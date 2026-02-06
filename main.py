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
    ideal_consec_work: int = 5          # soft (pośrednio przez kary na krótkie bloki)
    weekend_fairness_weight: int = 50   # soft
    shift_change_penalty: int = 10      # soft (gdy variety <= 4 lub jako kara rotacji)
    target_shifts_per_week: int = 5     # soft
    max_shifts_per_week: int = 6        # hard
    variety: int = 10                   # 0..10 (10=blokowo, 0=różnorodnie)


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


# ---------- HELPERY ----------
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
        return set()  # wymusza OFF (zakaz D/P/N)
    parts = [p.strip() for p in a.replace(";", ",").split(",") if p.strip()]
    return set([p for p in parts if p in SHIFT_KEYS])


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


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

    # variety 0..10
    V = clamp(int(cfg.variety), 0, 10)
    t = V / 10.0  # 0..1, 1=blokowo

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

    # work[e,d] (Bool) == 1 jeśli pracuje (czyli ma D/P/N)
    work = {}
    for e in range(nE):
        for d in range(nD):
            w = model.NewBoolVar(f"work_e{e}_d{d}")
            model.Add(sum(x[(e, d, s)] for s in range(3)) == w)
            work[(e, d)] = w

    # 1) Coverage (hard)
    for d in range(nD):
        dk = dow_key(dates[d])
        for s, sh_name in enumerate(SHIFT_KEYS):
            need = int(req.demand[dk][sh_name])
            model.Add(sum(x[(e, d, s)] for e in range(nE)) == need)

    # 2) Unavailability (hard)
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

    # 5) HARD: Legal rest constraints (11h rest) for day-to-day transitions
    # D: 06-14, P: 14-22, N: 22-06 (spans to next day)
    for e in range(nE):
        for d in range(1, nD):
            # P(prev) -> D(curr)
            model.Add(x[(e, d - 1, 1)] + x[(e, d, 0)] <= 1)
            # N(prev) -> P(curr)
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 1)] <= 1)
            # N(prev) -> D(curr)
            model.Add(x[(e, d - 1, 2)] + x[(e, d, 0)] <= 1)

    # ============================================================
    # FAIRNESS na liczbę DNI PRACY (hard): base..base+1
    # + dokładnie rem_work osób ma base+1
    # ============================================================
    total_work_required = 0
    for d in range(nD):
        dk = dow_key(dates[d])
        total_work_required += int(req.demand[dk]["D"]) + int(req.demand[dk]["P"]) + int(req.demand[dk]["N"])

    base_work = total_work_required // nE
    rem_work = total_work_required % nE

    work_count = []
    for e in range(nE):
        wc = model.NewIntVar(0, nD, f"workcnt_e{e}")
        model.Add(wc == sum(work[(e, d)] for d in range(nD)))
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

    # ============================================================
    # SHIFT-BLOCK / VARIETY MODE:
    # - V >= 5: HARD: jeśli pracuje dzień po dniu -> ta sama zmiana (blokowo)
    # - V <= 4: SOFT: kara za zmianę zmiany dzień-do-dnia (bardziej różnorodnie)
    # ============================================================
    hard_same_shift_blocks = (V >= 5)

    if hard_same_shift_blocks:
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

    # ----------------------------
    # FAIRNESS: rozkład D/P/N na pracownika (luźne hard + soft)
    # ----------------------------
    total_by_shift = [0, 0, 0]  # D,P,N
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

    # ----------------------------
    # WEEKEND fairness (soft)
    # ----------------------------
    sat_indices = [d for d in range(nD) if dates[d].weekday() == 5]
    sun_indices = [d for d in range(nD) if dates[d].weekday() == 6]

    sat_work = []
    sun_work = []
    for e in range(nE):
        sv = model.NewIntVar(0, len(sat_indices), f"sat_e{e}")
        uv = model.NewIntVar(0, len(sun_indices), f"sun_e{e}")
        model.Add(sv == sum(work[(e, dd)] for dd in sat_indices))
        model.Add(uv == sum(work[(e, dd)] for dd in sun_indices))
        sat_work.append(sv)
        sun_work.append(uv)

    # ----------------------------
    # OBJECTIVE (soft) — wagi zależne od variety
    # ----------------------------
    obj_terms = []

    # Weekend spread weight from cfg
    w_weekend = max(1, int(cfg.weekend_fairness_weight))

    # Bazowe (niezależne)
    w_work_weekend_day = 5
    w_split_weekend = 80

    # Zależne od variety (t=1 blokowo, t=0 różnorodnie)
    w_block_start = int(10 + 60 * t)       # starty bloków pracy
    w_single_work = int(20 + 140 * t)      # OFF-W-OFF
    w_two_work = int(10 + 90 * t)          # OFF-W-W-OFF

    # klucz: shiftA - OFF - shiftB
    w_change_sep_off = int(5 + 180 * t)

    # równowaga D/P/N
    w_shift_balance = int(10 + 20 * t)

    # spread dni pracy (już hard base..base+1, ale to dociska wybór kto ma +1)
    w_work_spread = 50

    # A) WORK - OFF - WORK (pojedynczy OFF)
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

    # B) shiftA - OFF - shiftB (A!=B)
    if nD >= 3:
        for e in range(nE):
            for d in range(1, nD - 1):
                for s in range(3):
                    for tt in range(3):
                        if s == tt:
                            continue
                        y = model.NewBoolVar(f"chgsep_e{e}_d{d}_s{s}_t{tt}")
                        model.Add(y <= x[(e, d - 1, s)])
                        model.Add(y <= 1 - work[(e, d)])
                        model.Add(y <= x[(e, d + 1, tt)])
                        model.Add(y >= x[(e, d - 1, s)] + (1 - work[(e, d)]) + x[(e, d + 1, tt)] - 2)
                        obj_terms.append(w_change_sep_off * y)

    # C) weekend split + small discourage weekend work
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
            model.Add(split == work[(e, sat)] + work[(e, sun)] - 2 * both)
            obj_terms.append(w_split_weekend * split)
            obj_terms.append(w_work_weekend_day * work[(e, sat)])
            obj_terms.append(w_work_weekend_day * work[(e, sun)])

    # D) minimize spread Sat/Sun counts (soft)
    if sat_indices:
        sat_max = model.NewIntVar(0, len(sat_indices), "sat_max")
        sat_min = model.NewIntVar(0, len(sat_indices), "sat_min")
        model.AddMaxEquality(sat_max, sat_work)
        model.AddMinEquality(sat_min, sat_work)
        sat_diff = model.NewIntVar(0, len(sat_indices), "sat_diff")
        model.Add(sat_diff == sat_max - sat_min)
        obj_terms.append(w_weekend * sat_diff)

    if sun_indices:
        sun_max = model.NewIntVar(0, len(sun_indices), "sun_max")
        sun_min = model.NewIntVar(0, len(sun_indices), "sun_min")
        model.AddMaxEquality(sun_max, sun_work)
        model.AddMinEquality(sun_min, sun_work)
        sun_diff = model.NewIntVar(0, len(sun_indices), "sun_diff")
        model.Add(sun_diff == sun_max - sun_min)
        obj_terms.append(w_weekend * sun_diff)

    # E) weekly target (soft) — waga też zależna od variety (blokowo = mocniej)
    target = int(cfg.target_shifts_per_week)
    w_weekly = int(5 + 20 * t)
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
                obj_terms.append(w_weekly * dev)

    # F) minimalizuj liczbę startów bloków pracy (dłuższe ciągi)
    for e in range(nE):
        obj_terms.append(w_block_start * work[(e, 0)])
        for d in range(1, nD):
            start_blk = model.NewBoolVar(f"startblk_e{e}_d{d}")
            model.Add(start_blk <= work[(e, d)])
            model.Add(start_blk <= 1 - work[(e, d - 1)])
            model.Add(start_blk >= work[(e, d)] - work[(e, d - 1)])
            obj_terms.append(w_block_start * start_blk)

    # G) karz krótkie bloki pracy: OFF-W-OFF i OFF-W-W-OFF
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

    # H) soft równoważenie D/P/N względem base
    for e in range(nE):
        for s in range(3):
            vcnt = shift_counts[(e, s)]
            tgt = bases[s]
            dev = model.NewIntVar(0, nD, f"dev_cnt_e{e}_s{s}")
            model.AddAbsEquality(dev, vcnt - tgt)
            obj_terms.append(w_shift_balance * dev)

    # I) soft: dociskaj różnicę (max-min) dni pracy (wybór kto ma +1)
    work_max = model.NewIntVar(0, nD, "work_max")
    work_min = model.NewIntVar(0, nD, "work_min")
    model.AddMaxEquality(work_max, work_count)
    model.AddMinEquality(work_min, work_count)
    work_diff = model.NewIntVar(0, nD, "work_diff")
    model.Add(work_diff == work_max - work_min)
    obj_terms.append(w_work_spread * work_diff)

    # J) Jeśli variety <= 4: dodaj soft karę za zmianę zmiany dzień-do-dnia
    # (bo nie mamy wtedy hard constraintu "same shift in blocks")
    if not hard_same_shift_blocks:
        base_pen = max(0, int(cfg.shift_change_penalty))
        # im większa variety (w tym przedziale 0..4), tym większa kara
        # V=0 => niska kara, V=4 => większa kara
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

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
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
