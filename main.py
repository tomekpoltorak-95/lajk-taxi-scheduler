from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, date
from ortools.sat.python import cp_model

app = FastAPI()


# ---------- MODELE ----------
class Config(BaseModel):
    weeks: int = 8
    max_consec_work: int = 6            # hard
    ideal_consec_work: int = 5          # soft (unused in constraints, only penalty)
    weekend_fairness_weight: int = 10
    shift_change_penalty: int = 3
    target_shifts_per_week: int = 5     # soft
    max_shifts_per_week: int = 6        # hard


class UnavailabilityItem(BaseModel):
    date: str       # "YYYY-MM-DD"
    name: str
    allowed: str = ""  # "OFF" or "D,P" etc


class SolveRequest(BaseModel):
    names: List[str]
    dates: List[str]  # "YYYY-MM-DD"
    demand: Dict[str, Dict[str, int]]  # e.g. {"MONDAY":{"D":2,"P":2,"N":1}, ...}
    config: Config
    unavailability: Optional[List[UnavailabilityItem]] = []


class SolveResponse(BaseModel):
    names: List[str]
    dates: List[str]
    matrix: List[List[str]]  # "D"/"P"/"N"/"OFF"


@app.get("/")
def root():
    return {"status": "ok", "service": "lajk-taxi-scheduler"}


# ---------- HELPERS ----------
DOW_KEYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
SHIFT_KEYS = ["D", "P", "N"]


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def dow_key(d: date) -> str:
    # Monday=0 ... Sunday=6
    return DOW_KEYS[d.weekday()]


def allowed_set(allowed: str) -> set:
    a = (allowed or "").strip().upper()
    if not a:
        return set(SHIFT_KEYS)  # no restriction
    if a == "OFF":
        return set()            # nothing allowed
    parts = [p.strip() for p in a.replace(";", ",").split(",") if p.strip()]
    return set([p for p in parts if p in SHIFT_KEYS])


# ---------- SOLVER ----------
@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    names = req.names
    dates_s = req.dates
    if not names or not dates_s:
        raise HTTPException(status_code=400, detail="names/dates required")

    cfg = req.config
    nE = len(names)
    nD = len(dates_s)

    dates = [parse_date(s) for s in dates_s]

    # Demand sanity
    for k in DOW_KEYS:
        if k not in req.demand:
            raise HTTPException(status_code=400, detail=f"Missing demand for {k}")
        for sh in SHIFT_KEYS:
            if sh not in req.demand[k]:
                raise HTTPException(status_code=400, detail=f"Missing demand[{k}][{sh}]")

    # Build unavailability map: (name, date)->allowed set
    ua = {}
    for item in (req.unavailability or []):
        ua[(item.name.strip(), item.date.strip())] = allowed_set(item.allowed)

    model = cp_model.CpModel()

    # x[e,d,s] binary: employee e works shift s on day d
    x = {}
    for e in range(nE):
        for d in range(nD):
            for s in range(3):
                x[(e, d, s)] = model.NewBoolVar(f"x_e{e}_d{d}_s{s}")

    # work[e,d] = 1 if works any shift that day
    work = {}
    for e in range(nE):
        for d in range(nD):
            w = model.NewBoolVar(f"work_e{e}_d{d}")
            model.Add(sum(x[(e, d, s)] for s in range(3)) == w)
            work[(e, d)] = w

    # 1) Coverage constraints
    for d in range(nD):
        dk = dow_key(dates[d])
        for s, sh_name in enumerate(SHIFT_KEYS):
            need = int(req.demand[dk][sh_name])
            model.Add(sum(x[(e, d, s)] for e in range(nE)) == need)

    # 2) Unavailability constraints
    for e in range(nE):
        name = names[e].strip()
        for d in range(nD):
            ds = dates_s[d]
            allowed = ua.get((name, ds), None)
            if allowed is None:
                continue
            # If allowed empty => OFF only => no shifts
            for s, sh_name in enumerate(SHIFT_KEYS):
                if sh_name not in allowed:
                    model.Add(x[(e, d, s)] == 0)

    # 3) Hard: no N -> D next day
    # N is s=2, D is s=0
    for e in range(nE):
        for d in range(1, nD):
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 0)] <= 1)

    # 4) Hard: max consecutive work <= cfg.max_consec_work
    max_consec = int(cfg.max_consec_work)
    if max_consec < 1:
        max_consec = 1
    window = max_consec + 1
    for e in range(nE):
        for start in range(0, nD - window + 1):
            model.Add(sum(work[(e, d)] for d in range(start, start + window)) <= max_consec)

    # 5) Hard: max shifts per week <= cfg.max_shifts_per_week
    max_week = int(cfg.max_shifts_per_week)
    weeks = int(cfg.weeks)
    for e in range(nE):
        for w in range(weeks):
            start = w * 7
            end = min(start + 7, nD)
            if start >= nD:
                break
            model.Add(sum(work[(e, d)] for d in range(start, end)) <= max_week)

    # ---------- OBJECTIVE (soft) ----------
    obj_terms = []

    # A) Penalize shift changes day-to-day (only when working both days)
    # Create transition variables y[e,d,s,t] for d>=1 and s!=t
    penalty_change = int(cfg.shift_change_penalty)
    if penalty_change > 0:
        for e in range(nE):
            for d in range(1, nD):
                for s in range(3):
                    for t in range(3):
                        if s == t:
                            continue
                        y = model.NewBoolVar(f"chg_e{e}_d{d}_s{s}_t{t}")
                        model.Add(y <= x[(e, d - 1, s)])
                        model.Add(y <= x[(e, d, t)])
                        model.Add(y >= x[(e, d - 1, s)] + x[(e, d, t)] - 1)
                        obj_terms.append(penalty_change * y)

    # B) Soft: prefer ~target shifts per week (minimize absolute deviation)
    target = int(cfg.target_shifts_per_week)
    if target > 0:
        for e in range(nE):
            for w in range(weeks):
                start = w * 7
                end = min(start + 7, nD)
                if start >= nD:
                    break
                wk_shifts = model.NewIntVar(0, 7, f"wksh_e{e}_w{w}")
                model.Add(wk_shifts == sum(work[(e, d)] for d in range(start, end)))

                dev = model.NewIntVar(0, 7, f"dev_e{e}_w{w}")
                model.AddAbsEquality(dev, wk_shifts - target)
                obj_terms.append(1 * dev)

    # C) Weekend fairness: minimize max-min weekend_worked across employees
    # Define weekend index w (Sat/Sun of each week)
    w_weight = int(cfg.weekend_fairness_weight)
    if w_weight > 0:
        weekend_worked = {}
        for e in range(nE):
            for w in range(weeks):
                sat = w * 7 + 5  # Monday=0 => Saturday index 5
                sun = w * 7 + 6
                if sat >= nD:
                    continue
                ww = model.NewBoolVar(f"wknd_e{e}_w{w}")
                sat_work = work[(e, sat)]
                if sun < nD:
                    sun_work = work[(e, sun)]
                    model.Add(ww >= sat_work)
                    model.Add(ww >= sun_work)
                    model.Add(ww <= sat_work + sun_work)
                else:
                    model.Add(ww == sat_work)
                weekend_worked[(e, w)] = ww

        # total weekend worked per employee
        wk_totals = []
        for e in range(nE):
            t = model.NewIntVar(0, weeks, f"wkndtot_e{e}")
            model.Add(t == sum(weekend_worked.get((e, w), 0) for w in range(weeks)))
            wk_totals.append(t)

        wk_max = model.NewIntVar(0, weeks, "wknd_max")
        wk_min = model.NewIntVar(0, weeks, "wknd_min")
        model.AddMaxEquality(wk_max, wk_totals)
        model.AddMinEquality(wk_min, wk_totals)

        diff = model.NewIntVar(0, weeks, "wknd_diff")
        model.Add(diff == wk_max - wk_min)
        obj_terms.append(w_weight * diff)

    # Solve
    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(status_code=400, detail="No feasible schedule found. Try more staff or relax constraints.")

    # Build matrix
    matrix = []
    for e in range(nE):
        row = []
        for d in range(nD):
            val = "OFF"
            if solver.Value(x[(e, d, 0)]) == 1:
                val = "D"
            elif solver.Value(x[(e, d, 1)]) == 1:
                val = "P"
            elif solver.Value(x[(e, d, 2)]) == 1:
                val = "N"
            row.append(val)
        matrix.append(row)

    return SolveResponse(names=names, dates=dates_s, matrix=matrix)
