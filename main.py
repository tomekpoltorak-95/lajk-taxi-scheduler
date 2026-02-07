from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, date
from ortools.sat.python import cp_model

app = FastAPI()

# ---------- MODELE ----------
class Config(BaseModel):
    weeks: int = 8
    max_consec_work: int = 6
    ideal_consec_work: int = 5
    weekend_fairness_weight: int = 50
    shift_change_penalty: int = 10
    target_shifts_per_week: int = 5
    max_shifts_per_week: int = 6
    variety: int = 10  # 0..10


class UnavailabilityItem(BaseModel):
    date: str
    name: str
    allowed: str = ""  # "OFF" or "D,P" etc (opcjonalnie też "S")


class SolveRequest(BaseModel):
    names: List[str]
    dates: List[str]
    demand: Dict[str, Dict[str, int]]  # teraz też może zawierać "S"
    config: Config
    unavailability: Optional[List[UnavailabilityItem]] = []


class SolveResponse(BaseModel):
    names: List[str]
    dates: List[str]
    matrix: List[List[str]]  # "D"/"P"/"N"/"S"/"OFF"


@app.get("/")
def root():
    return {"status": "ok", "service": "lajk-taxi-scheduler"}


# ---------- HELPERY ----------
DOW_KEYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
DRIVE_SHIFTS = ["D", "P", "N"]   # jazda
ONCALL_SHIFT = "S"              # on-call


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def dow_key(d: date) -> str:
    return DOW_KEYS[d.weekday()]


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def allowed_set(allowed: str) -> set:
    """
    allowed może zawierać: D,P,N,S albo OFF.
    - "" => dozwolone wszystko (D,P,N,S)
    - "OFF" => nic (wymusza OFF od wszystkiego)
    """
    a = (allowed or "").strip().upper()
    if not a:
        return set(DRIVE_SHIFTS + [ONCALL_SHIFT])
    if a == "OFF":
        return set()
    parts = [p.strip() for p in a.replace(";", ",").split(",") if p.strip()]
    out = set()
    for p in parts:
        if p in DRIVE_SHIFTS or p == ONCALL_SHIFT:
            out.add(p)
    return out


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

    V = clamp(int(cfg.variety), 0, 10)
    t = V / 10.0  # 0..1 (1=blokowo)

    dates = [parse_date(s) for s in dates_s]

    # Demand sanity (D,P,N wymagane; S opcjonalne -> default 0)
    for k in DOW_KEYS:
        if k not in req.demand:
            raise HTTPException(status_code=400, detail=f"Missing demand for {k}")
        for sh in DRIVE_SHIFTS:
            if sh not in req.demand[k]:
                raise HTTPException(status_code=400, detail=f"Missing demand[{k}][{sh}]")
        # S może nie istnieć w starych arkuszach
        if "S" not in req.demand[k]:
            req.demand[k]["S"] = 0

    # Unavailability map
    ua = {}
    for item in (req.unavailability or []):
        ua[(item.name.strip(), item.date.strip())] = allowed_set(item.allowed)

    model = cp_model.CpModel()

    # x[e,d,s] dla D/P/N
    x = {}
    for e in range(nE):
        for d in range(nD):
            for s in range(3):
                x[(e, d, s)] = model.NewBoolVar(f"x_e{e}_d{d}_s{s}")

    # y[e,d] dla S (on-call)
    y = {}
    for e in range(nE):
        for d in range(nD):
            y[(e, d)] = model.NewBoolVar(f"s_e{e}_d{d}")

    # work[e,d] (Bool) = czy pracuje D/P/N (S się NIE liczy jako praca)
    work = {}
    for e in range(nE):
        for d in range(nD):
            w = model.NewBoolVar(f"work_e{e}_d{d}")
            model.Add(sum(x[(e, d, s)] for s in range(3)) == w)
            work[(e, d)] = w

    # 0) S tylko gdy brak D/P/N w tym dniu (żeby nie było "D+S" w jednej komórce)
    for e in range(nE):
        for d in range(nD):
            model.Add(y[(e, d)] <= 1 - work[(e, d)])

    # 1) Coverage (hard) dla D/P/N + S
    for d in range(nD):
        dk = dow_key(dates[d])

        # D/P/N
        for s, sh_name in enumerate(DRIVE_SHIFTS):
            need = int(req.demand[dk][sh_name])
            model.Add(sum(x[(e, d, s)] for e in range(nE)) == need)

        # S (on-call)
        need_s = int(req.demand[dk].get("S", 0))
        model.Add(sum(y[(e, d)] for e in range(nE)) == need_s)

    # 2) Unavailability (hard) – obejmuje też S jeśli ktoś wpisze allowed
    for e in range(nE):
        name = names[e].strip()
        for d in range(nD):
            ds = dates_s[d]
            allowed = ua.get((name, ds), None)
            if allowed is None:
                continue

            # D/P/N
            for s, sh_name in enumerate(DRIVE_SHIFTS):
                if sh_name not in allowed:
                    model.Add(x[(e, d, s)] == 0)

            # S
            if ONCALL_SHIFT not in allowed:
                model.Add(y[(e, d)] == 0)

    # 3) Hard: max consecutive WORK (D/P/N)
    max_consec = max(1, int(cfg.max_consec_work))
    window = max_consec + 1
    for e in range(nE):
        for start in range(0, nD - window + 1):
            model.Add(sum(work[(e, dd)] for dd in range(start, start + window)) <= max_consec)

    # 4) Hard: max shifts/week (liczymy tylko D/P/N)
    max_week = int(cfg.max_shifts_per_week)
    for e in range(nE):
        for w in range(weeks):
            start = w * 7
            end = min(start + 7, nD)
            if start >= nD:
                break
            model.Add(sum(work[(e, dd)] for dd in range(start, end)) <= max_week)

    # 5) HARD: Legal rest constraints (D/P/N)
    for e in range(nE):
        for d in range(1, nD):
            # P(prev)->D(curr)
            model.Add(x[(e, d - 1, 1)] + x[(e, d, 0)] <= 1)
            # N(prev)->P(curr)
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 1)] <= 1)
            # N(prev)->D(curr)
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 0)] <= 1)

    # FAIRNESS: równe dni pracy (D/P/N) – hard base..base+1 + dokładnie rem osób ma +1
    total_work_required = 0
    for d in range(nD):
        dk = dow_key(dates[d])
        total_work_required += int(req.demand[dk]["D"]) + int(req.demand[dk]["P"]) + int(req.demand[dk]["N"])

    base_work = total_work_required // nE
    rem_work = total_work_required % nE

    work_count = []
    for e in range(nE):
        wc = model.NewIntVar(0, nD, f"workcnt_e{e}")
        model.Add(wc == sum(work[(e, dd)] for dd in range(nD)))
        work_count.append(wc)
        model.Add(wc >= base_work)
        model.Add(wc <= base_work + 1)

    if rem_work != 0:
        plus1_flags = []
        for e in range(nE):
            f = model.NewBoolVar(f"is_plus1_e{e}")
            model.Add(work_count[e] == base_work + 1).OnlyEnforceIf(f)
            model.Add(work_count[e] == base_work).OnlyEnforceIf(f.Not())
            plus1_flags.append(f)
        model.Add(sum(plus1_flags) == rem_work)

    # SHIFT STYLE / variety
    hard_same_shift_blocks = (V >= 5)
    obj_terms = []

    if hard_same_shift_blocks:
        # HARD: jeśli pracuje dzień po dniu -> ta sama zmiana (D/P/N)
        for e in range(nE):
            for d in range(1, nD):
                for s in range(3):
                    for tt in range(3):
                        if s == tt:
                            continue
                        model.Add(
                            x[(e, d - 1, s)] + x[(e, d, tt)]
                            <= 1 + (1 - work[(e, d - 1)]) + (1 - work[(e, d)])
                        )
    else:
        # SOFT: kara za zmianę zmiany dzień-do-dnia
        base_pen = max(0, int(cfg.shift_change_penalty))
        w_day_change = int(base_pen * (1 + (V / 4.0))) if base_pen > 0 else 0
        if w_day_change > 0:
            for e in range(nE):
                for d in range(1, nD):
                    for s in range(3):
                        for tt in range(3):
                            if s == tt:
                                continue
                            ch = model.NewBoolVar(f"chgday_e{e}_d{d}_s{s}_t{tt}")
                            model.Add(ch <= x[(e, d - 1, s)])
                            model.Add(ch <= x[(e, d, tt)])
                            model.Add(ch >= x[(e, d - 1, s)] + x[(e, d, tt)] - 1)
                            obj_terms.append(w_day_change * ch)

    # FAIRNESS: rozkład D/P/N (luźne hard + soft)
    total_by_shift = [0, 0, 0]
    for d in range(nD):
        dk = dow_key(dates[d])
        total_by_shift[0] += int(req.demand[dk]["D"])
        total_by_shift[1] += int(req.demand[dk]["P"])
        total_by_shift[2] += int(req.demand[dk]["N"])

    bases = [total_by_shift[s] // nE for s in range(3)]
    shift_counts = {}
    for e in range(nE):
        for s in range(3):
            vcnt = model.NewIntVar(0, nD, f"cnt_e{e}_s{s}")
            model.Add(vcnt == sum(x[(e, d, s)] for d in range(nD)))
            shift_counts[(e, s)] = vcnt

            low = max(0, bases[s] - 1)
            high = min(nD, bases[s] + 2)
            model.Add(vcnt >= low)
            model.Add(vcnt <= high)

    # FAIRNESS S (opcjonalnie, ale polecam): równo rozdziel S wśród pracowników
    # S nie liczy się do godzin, ale i tak lepiej rozłożyć równo “pod telefonem”.
    total_s_required = 0
    for d in range(nD):
        dk = dow_key(dates[d])
        total_s_required += int(req.demand[dk].get("S", 0))

    if total_s_required > 0:
        base_s = total_s_required // nE
        rem_s = total_s_required % nE
        s_counts = []
        for e in range(nE):
            sc = model.NewIntVar(0, nD, f"scnt_e{e}")
            model.Add(sc == sum(y[(e, d)] for d in range(nD)))
            s_counts.append(sc)
            model.Add(sc >= base_s)
            model.Add(sc <= base_s + 1)

        # soft: minimalizuj spread S
        w_s_spread = 20
        s_max = model.NewIntVar(0, nD, "s_max")
        s_min = model.NewIntVar(0, nD, "s_min")
        model.AddMaxEquality(s_max, s_counts)
        model.AddMinEquality(s_min, s_counts)
        s_diff = model.NewIntVar(0, nD, "s_diff")
        model.Add(s_diff == s_max - s_min)
        obj_terms.append(w_s_spread * s_diff)

    # WEEKEND fairness (soft) – liczymy weekendy tylko dla pracy D/P/N
    w_weekend = max(1, int(cfg.weekend_fairness_weight))
    sat_indices = [d for d in range(nD) if dates[d].weekday() == 5]
    sun_indices = [d for d in range(nD) if dates[d].weekday() == 6]

    if sat_indices:
        sat_work = []
        for e in range(nE):
            sv = model.NewIntVar(0, len(sat_indices), f"sat_e{e}")
            model.Add(sv == sum(work[(e, dd)] for dd in sat_indices))
            sat_work.append(sv)
        sat_max = model.NewIntVar(0, len(sat_indices), "sat_max")
        sat_min = model.NewIntVar(0, len(sat_indices), "sat_min")
        model.AddMaxEquality(sat_max, sat_work)
        model.AddMinEquality(sat_min, sat_work)
        sat_diff = model.NewIntVar(0, len(sat_indices), "sat_diff")
        model.Add(sat_diff == sat_max - sat_min)
        obj_terms.append(w_weekend * sat_diff)

    if sun_indices:
        sun_work = []
        for e in range(nE):
            uv = model.NewIntVar(0, len(sun_indices), f"sun_e{e}")
            model.Add(uv == sum(work[(e, dd)] for dd in sun_indices))
            sun_work.append(uv)
        sun_max = model.NewIntVar(0, len(sun_indices), "sun_max")
        sun_min = model.NewIntVar(0, len(sun_indices), "sun_min")
        model.AddMaxEquality(sun_max, sun_work)
        model.AddMinEquality(sun_min, sun_work)
        sun_diff = model.NewIntVar(0, len(sun_indices), "sun_diff")
        model.Add(sun_diff == sun_max - sun_min)
        obj_terms.append(w_weekend * sun_diff)

    # --- Objective: styl blokowy vs luźny (dotyczy D/P/N)
    w_block_start = int(10 + 60 * t)
    w_single_work = int(20 + 140 * t)
    w_two_work = int(10 + 90 * t)
    w_change_sep_off = int(5 + 180 * t)
    w_shift_balance = int(10 + 20 * t)

    # A) WORK-OFF-WORK
    w_single_off = 50
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                z = model.NewBoolVar(f"singleoff_e{e}_d{d}")
                model.Add(z <= work[(e, d - 1)])
                model.Add(z <= work[(e, d + 1)])
                model.Add(z <= 1 - work[(e, d)])
                model.Add(z >= work[(e, d - 1)] + work[(e, d + 1)] + (1 - work[(e, d)]) - 2)
                obj_terms.append(w_single_off * z)

    # B) shiftA-OFF-shiftB (A!=B)
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                for s in range(3):
                    for tt in range(3):
                        if s == tt:
                            continue
                        yv = model.NewBoolVar(f"chgsep_e{e}_d{d}_s{s}_t{tt}")
                        model.Add(yv <= x[(e, d - 1, s)])
                        model.Add(yv <= 1 - work[(e, d)])
                        model.Add(yv <= x[(e, d + 1, tt)])
                        model.Add(yv >= x[(e, d - 1, s)] + (1 - work[(e, d)]) + x[(e, d + 1, tt)] - 2)
                        obj_terms.append(w_change_sep_off * yv)

    # C) minimalizuj starty bloków pracy
    for e in range(nE):
        obj_terms.append(w_block_start * work[(e, 0)])
        for d in range(1, nD):
            start_blk = model.NewBoolVar(f"startblk_e{e}_d{d}")
            model.Add(start_blk <= work[(e, d)])
            model.Add(start_blk <= 1 - work[(e, d - 1)])
            model.Add(start_blk >= work[(e, d)] - work[(e, d - 1)])
            obj_terms.append(w_block_start * start_blk)

    # D) karz OFF-W-OFF i OFF-W-W-OFF
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                one = model.NewBoolVar(f"singlework_e{e}_d{d}")
                model.Add(one <= 1 - work[(e, d - 1)])
                model.Add(one <= work[(e, d)])
                model.Add(one <= 1 - work[(e, d + 1)])
                model.Add(one >= (1 - work[(e, d - 1)]) + work[(e, d)] + (1 - work[(e, d + 1)]) - 2)
                obj_terms.append(w_single_work * one)

    if nD >= 4:
        for e in range(nE):
            for d in range(1, nD - 2):
                two = model.NewBoolVar(f"twowork_e{e}_d{d}")
                model.Add(two <= 1 - work[(e, d - 1)])
                model.Add(two <= work[(e, d)])
                model.Add(two <= work[(e, d + 1)])
                model.Add(two <= 1 - work[(e, d + 2)])
                model.Add(two >= (1 - work[(e, d - 1)]) + work[(e, d)] + work[(e, d + 1)] + (1 - work[(e, d + 2)]) - 3)
                obj_terms.append(w_two_work * two)

    # E) soft równowaga D/P/N
    for e in range(nE):
        for s in range(3):
            vcnt = shift_counts[(e, s)]
            tgt = bases[s]
            dev = model.NewIntVar(0, nD, f"dev_cnt_e{e}_s{s}")
            model.AddAbsEquality(dev, vcnt - tgt)
            obj_terms.append(w_shift_balance * dev)

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HTTPException(status_code=400, detail="No feasible schedule found. Try more staff or relax constraints.")

    # Build matrix: D/P/N ma priorytet, potem S, potem OFF
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
            elif solver.Value(y[(e, d)]) == 1:
                val = "S"
            row.append(val)
        matrix.append(row)

    return SolveResponse(names=names, dates=dates_s, matrix=matrix)
