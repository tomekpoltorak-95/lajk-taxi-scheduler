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
    ideal_consec_work: int = 5          # soft
    weekend_fairness_weight: int = 10   # used as weights in objective (not hard weekend fairness)
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
    demand: Dict[str, Dict[str, int]]  # {"MONDAY":{"D":2,"P":2,"N":1}, ...}
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
    return DOW_KEYS[d.weekday()]  # Monday=0..Sunday=6


def allowed_set(allowed: str) -> set:
    a = (allowed or "").strip().upper()
    if not a:
        return set(SHIFT_KEYS)
    if a == "OFF":
        return set()
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
    weeks = int(cfg.weeks)

    dates = [parse_date(s) for s in dates_s]

    # Demand sanity
    for k in DOW_KEYS:
        if k not in req.demand:
            raise HTTPException(status_code=400, detail=f"Missing demand for {k}")
        for sh in SHIFT_KEYS:
            if sh not in req.demand[k]:
                raise HTTPException(status_code=400, detail=f"Missing demand[{k}][{sh}]")

    # Unavailability map: (name, date)->allowed set
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

    # work[e,d] = 1 if works any shift that day (else OFF)
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
            for s, sh_name in enumerate(SHIFT_KEYS):
                if sh_name not in allowed:
                    model.Add(x[(e, d, s)] == 0)

    # 3) Hard: no N -> D next day (N is s=2, D is s=0)
    for e in range(nE):
        for d in range(1, nD):
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 0)] <= 1)

    # 4) Hard: max consecutive work <= cfg.max_consec_work
    max_consec = max(1, int(cfg.max_consec_work))
    window = max_consec + 1
    for e in range(nE):
        for start in range(0, nD - window + 1):
            model.Add(sum(work[(e, d)] for d in range(start, start + window)) <= max_consec)

    # 5) Hard: max shifts per week <= cfg.max_shifts_per_week
    max_week = int(cfg.max_shifts_per_week)
    for e in range(nE):
        for w in range(weeks):
            start = w * 7
            end = min(start + 7, nD)
            if start >= nD:
                break
            model.Add(sum(work[(e, d)] for d in range(start, end)) <= max_week)

    # ==========================================================
    # 6) HARD: Weekend fairness (days Sat+Sun across whole horizon)
    #    This prevents the crazy imbalance you saw.
    #
    #    We count weekend DAYS worked (not boolean "weekend used").
    #    Total weekend work-days is fixed by demand; we split it fairly.
    # ==========================================================
    weekend_day_indices = []
    for d in range(nD):
        wd = dates[d].weekday()  # Mon=0..Sun=6
        if wd in (5, 6):         # Sat=5, Sun=6
            weekend_day_indices.append(d)

    total_weekend_days_required = 0
    # compute required weekend day staffing from demand
    for d in weekend_day_indices:
        dk = dow_key(dates[d])
        total_weekend_days_required += int(req.demand[dk]["D"]) + int(req.demand[dk]["P"]) + int(req.demand[dk]["N"])

    # target distribution
    base = total_weekend_days_required // nE
    rem = total_weekend_days_required % nE
    # If divisible -> exact. Else -> [base, base+1]
    wk_targets = []
    for e in range(nE):
        wk_targets.append(base + (1 if e < rem else 0))

    # weekend_days_worked[e] == wk_targets[e] (or between base/base+1 if you prefer)
    weekend_days_worked = []
    for e in range(nE):
        v = model.NewIntVar(0, len(weekend_day_indices), f"wkdays_e{e}")
        model.Add(v == sum(work[(e, d)] for d in weekend_day_indices))
        weekend_days_worked.append(v)
        # HARD fairness:
        model.Add(v == wk_targets[e])

    # ---------- OBJECTIVE (soft) ----------
    obj_terms = []

    # Weights (stronger stability)
    change_pen = max(0, int(cfg.shift_change_penalty)) * 8  # much stronger than before
    single_off_pen = max(5, int(cfg.shift_change_penalty) * 6)
    short_run_1_pen = 20
    short_run_2_pen = 10
    split_weekend_pen = max(20, int(cfg.weekend_fairness_weight) * 4)

    # A) Penalize shift changes day-to-day (working both days, different shift)
    if change_pen > 0:
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
                        obj_terms.append(change_pen * y)

    # B) Penalize single OFF between working days: work(d-1)=1, work(d)=0, work(d+1)=1
    if single_off_pen > 0 and nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                z = model.NewBoolVar(f"singleoff_e{e}_d{d}")
                # z == 1 if work(d-1)=1 and work(d)=0 and work(d+1)=1
                model.Add(z <= work[(e, d - 1)])
                model.Add(z <= work[(e, d + 1)])
                model.Add(z <= 1 - work[(e, d)])
                model.Add(z >= work[(e, d - 1)] + work[(e, d + 1)] + (1 - work[(e, d)]) - 2)
                obj_terms.append(single_off_pen * z)

    # C) Prefer longer blocks of same shift:
    #    Penalize isolated 1-day run: s at d, but not s at d-1 and not s at d+1
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                for s in range(3):
                    iso = model.NewBoolVar(f"iso_e{e}_d{d}_s{s}")
                    model.Add(iso <= x[(e, d, s)])
                    model.Add(iso <= 1 - x[(e, d - 1, s)])
                    model.Add(iso <= 1 - x[(e, d + 1, s)])
                    model.Add(iso >= x[(e, d, s)] - x[(e, d - 1, s)] - x[(e, d + 1, s)])
                    obj_terms.append(short_run_1_pen * iso)

    # D) Penalize 2-day run surrounded by different (encourage >=3):
    #    x(d,s)=1 and x(d+1,s)=1 but x(d-1,s)=0 and x(d+2,s)=0
    if nD >= 4:
        for e in range(nE):
            for d in range(1, nD - 2):
                for s in range(3):
                    run2 = model.NewBoolVar(f"run2_e{e}_d{d}_s{s}")
                    model.Add(run2 <= x[(e, d, s)])
                    model.Add(run2 <= x[(e, d + 1, s)])
                    model.Add(run2 <= 1 - x[(e, d - 1, s)])
                    model.Add(run2 <= 1 - x[(e, d + 2, s)])
                    # lower bound
                    model.Add(run2 >= x[(e, d, s)] + x[(e, d + 1, s)] - x[(e, d - 1, s)] - x[(e, d + 2, s)] - 1)
                    obj_terms.append(short_run_2_pen * run2)

    # E) Weekend split penalty: works exactly one of (Sat,Sun) within the same week
    if split_weekend_pen > 0:
        for w in range(weeks):
            sat = w * 7 + 5
            sun = w * 7 + 6
            if sat >= nD:
                continue
            for e in range(nE):
                if sun >= nD:
                    continue
                split = model.NewBoolVar(f"splitwknd_e{e}_w{w}")
                # split == 1 if sat+sun == 1
                model.Add(split <= work[(e, sat)] + work[(e, sun)])
                model.Add(split <= 2 - (work[(e, sat)] + work[(e, sun)]))
                model.Add(split >= work[(e, sat)] + work[(e, sun)] - 1)
                obj_terms.append(split_weekend_pen * split)

    # F) Soft: keep weekly shifts near target (small weight now)
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

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(
            status_code=400,
            detail="No feasible schedule found. Try more staff or relax constraints."
        )

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
