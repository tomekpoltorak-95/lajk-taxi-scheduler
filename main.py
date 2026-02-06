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
    weekend_fairness_weight: int = 50   # strong soft weights
    shift_change_penalty: int = 10
    target_shifts_per_week: int = 5     # soft
    max_shifts_per_week: int = 6        # hard


class UnavailabilityItem(BaseModel):
    date: str
    name: str
    allowed: str = ""  # "OFF" or "D,P" etc


class SolveRequest(BaseModel):
    names: List[str]
    dates: List[str]  # "YYYY-MM-DD"
    demand: Dict[str, Dict[str, int]]
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
    return DOW_KEYS[d.weekday()]


def allowed_set(allowed: str) -> set:
    a = (allowed or "").strip().upper()
    if not a:
        return set(SHIFT_KEYS)
    if a == "OFF":
        return set()
    parts = [p.strip() for p in a.replace(";", ",").split(",") if p.strip()]
    return set([p for p in parts if p in SHIFT_KEYS])


def bounds_from_total(total_required: int, nE: int):
    base = total_required // nE
    rem = total_required % nE
    return base, rem


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

    # Unavailability map
    ua = {}
    for item in (req.unavailability or []):
        ua[(item.name.strip(), item.date.strip())] = allowed_set(item.allowed)

    model = cp_model.CpModel()

    # x[e,d,s]
    x = {}
    for e in range(nE):
        for d in range(nD):
            for s in range(3):
                x[(e, d, s)] = model.NewBoolVar(f"x_e{e}_d{d}_s{s}")

    # work[e,d]
    work = {}
    for e in range(nE):
        for d in range(nD):
            w = model.NewBoolVar(f"work_e{e}_d{d}")
            model.Add(sum(x[(e, d, s)] for s in range(3)) == w)
            work[(e, d)] = w

    # 1) Coverage
    for d in range(nD):
        dk = dow_key(dates[d])
        for s, sh_name in enumerate(SHIFT_KEYS):
            need = int(req.demand[dk][sh_name])
            model.Add(sum(x[(e, d, s)] for e in range(nE)) == need)

    # 2) Unavailability
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

    # 3) Hard: max consecutive work
    max_consec = max(1, int(cfg.max_consec_work))
    window = max_consec + 1
    for e in range(nE):
        for start in range(0, nD - window + 1):
            model.Add(sum(work[(e, d)] for d in range(start, start + window)) <= max_consec)

    # 4) Hard: max shifts/week
    max_week = int(cfg.max_shifts_per_week)
    for e in range(nE):
        for w in range(weeks):
            start = w * 7
            end = min(start + 7, nD)
            if start >= nD:
                break
            model.Add(sum(work[(e, d)] for d in range(start, end)) <= max_week)

    # ============================================================
    # 5) HARD: Legal rest constraints (11h rest) for day-to-day transitions
    # We treat assignment as: day d shift happens that day:
    # D: 06-14, P: 14-22, N: 22-06 (spans to next day)
    # Illegal next-day transitions:
    # - P -> D (22 -> 06 = 8h)
    # - N -> P (06 -> 14 = 8h)  [N ends next day 06]
    # - N -> D (06 -> 06 = 0h)  [already bad]
    # ============================================================
    for e in range(nE):
        for d in range(1, nD):
            # P(prev) -> D(curr)
            model.Add(x[(e, d - 1, 1)] + x[(e, d, 0)] <= 1)
            # N(prev) -> P(curr)
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 1)] <= 1)
            # N(prev) -> D(curr)
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 0)] <= 1)

    # ============================================================
    # 6) HARD: If employee works on consecutive days, they must keep SAME shift
    # (shift changes only allowed if at least one OFF day between blocks)
    # ============================================================
    for e in range(nE):
        for d in range(1, nD):
            for s in range(3):
                for t in range(3):
                    if s == t:
                        continue
                    # If both days are worked, forbid different shift pair
                    # x_prev_s + x_curr_t <= 1 + (1-work_prev) + (1-work_curr)
                    model.Add(
                        x[(e, d - 1, s)] + x[(e, d, t)]
                        <= 1 + (1 - work[(e, d - 1)]) + (1 - work[(e, d)])
                    )

    # ----------------------------
    # WEEKEND fairness SAT & SUN separately (hard bounds + soft spread)
    # ----------------------------
    sat_indices = [d for d in range(nD) if dates[d].weekday() == 5]
    sun_indices = [d for d in range(nD) if dates[d].weekday() == 6]

    def total_required_on(indices: List[int]) -> int:
        tot = 0
        for dd in indices:
            dk = dow_key(dates[dd])
            tot += int(req.demand[dk]["D"]) + int(req.demand[dk]["P"]) + int(req.demand[dk]["N"])
        return tot

    total_sat = total_required_on(sat_indices)
    total_sun = total_required_on(sun_indices)

    sat_base, _ = bounds_from_total(total_sat, nE)
    sun_base, _ = bounds_from_total(total_sun, nE)

    sat_work = []
    sun_work = []
    for e in range(nE):
        sv = model.NewIntVar(0, len(sat_indices), f"sat_e{e}")
        uv = model.NewIntVar(0, len(sun_indices), f"sun_e{e}")
        model.Add(sv == sum(work[(e, dd)] for dd in sat_indices))
        model.Add(uv == sum(work[(e, dd)] for dd in sun_indices))
        sat_work.append(sv)
        sun_work.append(uv)

        # hard bounds
        model.Add(sv >= sat_base)
        model.Add(sv <= sat_base + 1)
        model.Add(uv >= sun_base)
        model.Add(uv <= sun_base + 1)

        # ============================================================
    # FAIRNESS: rozkład D/P/N na pracownika (hard bounds base..base+1)
    # ============================================================
    total_by_shift = [0, 0, 0]  # D,P,N
    for d in range(nD):
        dk = dow_key(dates[d])
        total_by_shift[0] += int(req.demand[dk]["D"])
        total_by_shift[1] += int(req.demand[dk]["P"])
        total_by_shift[2] += int(req.demand[dk]["N"])

    bases = [total_by_shift[s] // nE for s in range(3)]
    rems  = [total_by_shift[s] %  nE for s in range(3)]

    # Liczniki per pracownik
    shift_counts = {}  # (e,s) -> IntVar
    for e in range(nE):
        for s in range(3):
            v = model.NewIntVar(0, nD, f"cnt_e{e}_s{s}")
            model.Add(v == sum(x[(e, d, s)] for d in range(nD)))
            shift_counts[(e, s)] = v

            # Twarde widełki base..base+1
            # (wystarczy, żeby rozkład był bardzo równy i bez patologii)
            model.Add(v >= bases[s])
            model.Add(v <= bases[s] + 1)


    # ----------------------------
    # OBJECTIVE (soft): blocks + nicer weekends
    # ----------------------------
    obj_terms = []

    # weights
    w_weekend = max(1, int(cfg.weekend_fairness_weight))
    w_single_off = 50
    w_change_sep_off = 120      # big penalty: shiftA - OFF - shiftB (A!=B)
    w_split_weekend = 80        # penalty: work exactly one day of weekend
    w_work_weekend_day = 5      # small penalty to encourage full weekends off when possible

    # A) Penalize single OFF between work days
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                z = model.NewBoolVar(f"singleoff_e{e}_d{d}")
                model.Add(z <= work[(e, d - 1)])
                model.Add(z <= work[(e, d + 1)])
                model.Add(z <= 1 - work[(e, d)])
                model.Add(z >= work[(e, d - 1)] + work[(e, d + 1)] + (1 - work[(e, d)]) - 2)
                obj_terms.append(w_single_off * z)

    # B) Penalize shift-change separated by ONE OFF:
    # shift s at d-1, OFF at d, shift t at d+1, s!=t
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                for s in range(3):
                    for t in range(3):
                        if s == t:
                            continue
                        y = model.NewBoolVar(f"chgsep_e{e}_d{d}_s{s}_t{t}")
                        model.Add(y <= x[(e, d - 1, s)])
                        model.Add(y <= 1 - work[(e, d)])
                        model.Add(y <= x[(e, d + 1, t)])
                        model.Add(y >= x[(e, d - 1, s)] + (1 - work[(e, d)]) + x[(e, d + 1, t)] - 2)
                        obj_terms.append(w_change_sep_off * y)

    # C) Weekend split penalty (XOR Sat/Sun in each week)
    for w in range(weeks):
        sat = w * 7 + 5
        sun = w * 7 + 6
        if sat >= nD or sun >= nD:
            continue
        for e in range(nE):
            both = model.NewBoolVar(f"bothwknd_e{e}_w{w}")
            model.Add(both <= work[(e, sat)])
            model.Add(both <= work[(e, sun)])
            model.Add(both >= work[(e, sat)] + work[(e, sun)] - 1)

            split = model.NewIntVar(0, 1, f"splitwknd_e{e}_w{w}")
            model.Add(split == work[(e, sat)] + work[(e, sun)] - 2 * both)  # 0,1,0
            obj_terms.append(w_split_weekend * split)

            # Encourage full weekends OFF when possible (small)
            obj_terms.append(w_work_weekend_day * work[(e, sat)])
            obj_terms.append(w_work_weekend_day * work[(e, sun)])

    # D) Minimize spread of Sat & Sun counts (extra)
    sat_max = model.NewIntVar(0, len(sat_indices), "sat_max")
    sat_min = model.NewIntVar(0, len(sat_indices), "sat_min")
    model.AddMaxEquality(sat_max, sat_work)
    model.AddMinEquality(sat_min, sat_work)
    sat_diff = model.NewIntVar(0, len(sat_indices), "sat_diff")
    model.Add(sat_diff == sat_max - sat_min)
    obj_terms.append(w_weekend * sat_diff)

    sun_max = model.NewIntVar(0, len(sun_indices), "sun_max")
    sun_min = model.NewIntVar(0, len(sun_indices), "sun_min")
    model.AddMaxEquality(sun_max, sun_work)
    model.AddMinEquality(sun_min, sun_work)
    sun_diff = model.NewIntVar(0, len(sun_indices), "sun_diff")
    model.Add(sun_diff == sun_max - sun_min)
    obj_terms.append(w_weekend * sun_diff)

    # E) Keep weekly shifts near target (light)
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
    solver.parameters.max_time_in_seconds = 30.0
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
