"""Build LLM context from Binary Ninja analysis (decompiler-first).

The primary representation is HLIL (Binary Ninja's decompiler output). Where HLIL
is unavailable or a mapping fails, we fall back to MLIL text and finally to
disassembly. Every Binary Ninja API access is wrapped defensively because IL
availability and address->IL mapping can both fail (per BN docs, call-site to
HLIL conversion sometimes produces no result).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import prototypes as _prototypes
from . import tokens as _tokens

try:  # Binary Ninja is only present inside the BN Python environment.
    import binaryninja  # noqa: F401
    from binaryninja import (
        DisassemblySettings,
        DisassemblyOption,
        LinearViewObject,
        LinearViewCursor,
    )
    _HAVE_BN = True
except Exception:  # pragma: no cover - allows importing for offline tests
    binaryninja = None  # type: ignore
    _HAVE_BN = False


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class ContextBundle:
    text: str
    representation: str = "hlil"
    call_paths_text: str = ""
    tainted_sources: List[str] = field(default_factory=list)
    truncated: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def function_names(func) -> List[str]:
    """Candidate names for a function: raw, symbol short/full, demangled."""
    names: List[str] = []
    n = _safe(lambda: func.name)
    if n:
        names.append(n)
    sym = _safe(lambda: func.symbol)
    if sym is not None:
        for attr in ("short_name", "full_name", "raw_name"):
            v = _safe(lambda a=attr: getattr(sym, a))
            if v and v not in names:
                names.append(v)
    return names


def _same_instr(a, b) -> bool:
    if a is None or b is None:
        return False
    ea = _safe(lambda: a.expr_index)
    eb = _safe(lambda: b.expr_index)
    if ea is not None and eb is not None:
        return ea == eb
    return a is b


def _op_name(instr) -> str:
    op = _safe(lambda: instr.operation)
    if op is None:
        return ""
    return _safe(lambda: op.name, "") or str(op)


def _get_callers(bv, func) -> List:
    callers = _safe(lambda: list(func.callers))
    if callers:
        return _unique_functions(callers)
    sites = _safe(lambda: list(func.caller_sites)) or []
    out = []
    for s in sites:
        f = _safe(lambda s=s: s.function)
        if f is not None:
            out.append(f)
    if out:
        return _unique_functions(out)
    refs = _safe(lambda: list(bv.get_code_refs(func.start))) or []
    for r in refs:
        f = _safe(lambda r=r: r.function)
        if f is not None:
            out.append(f)
    return _unique_functions(out)


def _unique_functions(funcs) -> List:
    seen = set()
    out = []
    for f in funcs:
        start = _safe(lambda f=f: f.start)
        if start is None or start in seen:
            continue
        seen.add(start)
        out.append(f)
    return out


def _entry_function_starts(bv) -> set:
    starts = set()
    ef = _safe(lambda: list(bv.entry_functions)) or []
    for f in ef:
        s = _safe(lambda f=f: f.start)
        if s is not None:
            starts.add(s)
    e = _safe(lambda: bv.entry_function)
    if e is not None:
        s = _safe(lambda: e.start)
        if s is not None:
            starts.add(s)
    return starts


# --------------------------------------------------------------------------- #
# Decompiler / disassembly text
# --------------------------------------------------------------------------- #

def _linear_hlil_text(func) -> str:
    # The language-representation linear view does not generate IL itself: if HLIL
    # isn't cached yet it renders the prototype plus a "Loading..." placeholder and
    # queues generation asynchronously. Force synchronous HLIL generation first.
    if _safe(lambda: func.hlil) is None:
        return ""
    settings = DisassemblySettings()
    _safe(lambda: settings.set_option(DisassemblyOption.ShowVariableTypesWhenAssigned, True))
    obj = LinearViewObject.single_function_language_representation(func, settings)
    cursor = LinearViewCursor(obj)
    _safe(lambda: cursor.seek_to_begin())
    lines: List[str] = []
    guard = 0
    while guard < 200000:
        guard += 1
        cur_lines = _safe(lambda: cursor.lines) or []
        for ln in cur_lines:
            lines.append(str(ln))
        if not _safe(lambda: cursor.next(), False):
            break
    return "\n".join(lines)


def _is_placeholder_text(text: str) -> bool:
    """True if the linear view only produced BN's async "Loading..." placeholder."""
    return any(ln.strip() == "Loading..." for ln in text.splitlines())


def _disassembly_text(func) -> str:
    lines: List[str] = []
    instrs = _safe(lambda: func.instructions) or []
    for item in instrs:
        try:
            toks, addr = item
            text = "".join(getattr(t, "text", str(t)) for t in toks)
            lines.append(f"{addr:#x}  {text}")
        except Exception:
            continue
    return "\n".join(lines)


def decompile_text(func):
    """Return ``(text, representation)`` using the best available representation."""
    if _HAVE_BN:
        text = _safe(lambda: _linear_hlil_text(func))
        if text and text.strip() and not _is_placeholder_text(text):
            return text, "hlil"
        hl = _safe(lambda: func.hlil)
        if hl is not None:
            t = _safe(lambda: str(hl))
            if t and t.strip():
                return t, "hlil"
        ml = _safe(lambda: func.mlil)
        if ml is not None:
            t = _safe(lambda: str(ml))
            if t and t.strip():
                return t, "mlil"
    dis = _safe(lambda: _disassembly_text(func))
    if dis and dis.strip():
        return dis, "disassembly"
    return "(decompilation unavailable)", "none"


# --------------------------------------------------------------------------- #
# Call-site analysis
# --------------------------------------------------------------------------- #

def hlil_instr_at(func, addr):
    """Best-effort HLIL instruction at an address."""
    llil = _safe(lambda: func.get_low_level_il_at(addr))
    if llil is not None:
        h = _safe(lambda: llil.hlil)
        if h is not None:
            return h
    hl = _safe(lambda: func.hlil)
    if hl is not None:
        instrs = _safe(lambda: list(hl.instructions)) or []
        for i in instrs:
            if _safe(lambda i=i: i.address) == addr:
                return i
    return None


def render_call(instr) -> str:
    if instr is None:
        return "(call expression not recoverable from HLIL)"
    text = _safe(lambda: str(instr), "")
    params = _safe(lambda: list(instr.params))
    if params:
        rendered = "; ".join(f"arg{i}={_safe(lambda p=p: str(p), '?')}" for i, p in enumerate(params))
        return f"{text}\n  arguments: {rendered}"
    return text


def arg_var_sizes_text(arg_vars) -> str:
    """Per-argument-variable declared type and byte size (buffer capacities)."""
    lines: List[str] = []
    for v in arg_vars or []:
        name = _safe(lambda v=v: v.name) or "?"
        t = _safe(lambda v=v: v.type)
        if t is None:
            lines.append(f"- {name}: (type unknown)")
            continue
        tstr = _safe(lambda t=t: str(t)) or "?"
        width = _safe(lambda t=t: t.width)
        elem = _safe(lambda t=t: t.element_type)
        count = _safe(lambda t=t: t.count)
        note = ""
        if elem is not None and isinstance(count, int):
            ew = _safe(lambda e=elem: e.width)
            note = f"; fixed-size array of {count} element(s)"
            if isinstance(ew, int):
                note += f" x {ew} byte(s)"
        if isinstance(width, int):
            lines.append(f"- {name}: {tstr}  -> {width} byte(s){note}")
        else:
            lines.append(f"- {name}: {tstr}{note}")
    if not lines:
        return ""
    return ("Declared sizes of the variables passed to the call (use these as the "
            "destination/source buffer capacities; for pointers to heap memory the "
            "capacity is the allocation size, traced in the data-flow sections):\n"
            + "\n".join(lines))


def _type_capacity_note(t) -> str:
    """Describe a type's buffer capacity, resolving arrays, struct fields and
    pointers-to-arrays/structs to a concrete byte size where possible."""
    if t is None:
        return "type unknown"
    tstr = _safe(lambda: str(t)) or "?"
    width = _safe(lambda: t.width)
    elem = _safe(lambda: t.element_type)
    count = _safe(lambda: t.count)
    if elem is not None and isinstance(count, int):
        if isinstance(width, int):
            return f"{tstr}  -> fixed-size array, capacity {width} byte(s)"
        return f"{tstr}  -> fixed-size array"
    target = _safe(lambda: t.target)
    if target is not None:
        t_elem = _safe(lambda: target.element_type)
        t_count = _safe(lambda: target.count)
        t_width = _safe(lambda: target.width)
        t_str = _safe(lambda: str(target)) or "?"
        if t_elem is not None and isinstance(t_count, int):
            cap = f", capacity {t_width} byte(s)" if isinstance(t_width, int) else ""
            return f"{tstr}  -> pointer to fixed-size array '{t_str}'{cap}"
        if isinstance(t_width, int) and t_width > 1:
            return f"{tstr}  -> pointer to '{t_str}' object of {t_width} byte(s)"
        return (f"{tstr}  -> pointer; pointed-to buffer size unknown (capacity is "
                "the allocation size, see the data-flow sections)")
    if isinstance(width, int):
        return f"{tstr}  -> {width} byte(s)"
    return tstr


def arg_buffer_sizes_text(instr) -> str:
    """Per-call-argument type and buffer capacity, derived from the HLIL
    expression types (so struct fields like ``s->buf`` resolve to their real
    array size rather than the pointer size of ``s``)."""
    if instr is None:
        return ""
    call_exprs = list(_iter_call_exprs(instr)) or [instr]
    lines: List[str] = []
    for ce in call_exprs:
        params = _safe(lambda ce=ce: list(ce.params)) or []
        for idx, p in enumerate(params):
            ptext = _safe(lambda p=p: str(p), "?")
            note = _type_capacity_note(_safe(lambda p=p: p.expr_type))
            lines.append(f"- arg{idx} ({ptext}): {note}")
    if not lines:
        return ""
    return ("Type and buffer capacity of each value passed to the call. For arrays "
            "and pointers-to-arrays/structs (including struct fields) the capacity is "
            "the real buffer/field size in bytes; for a bare pointer the destination "
            "capacity is the allocation size (trace it in the data-flow sections):\n"
            + "\n".join(lines))


def argument_provenance(func, instr, tainted_source_set) -> str:
    """Classify each call argument as constant / parameter / from-call / tainted /
    local. This is the single strongest signal for whether an argument is
    attacker-influenced, and the main lever for avoiding false positives: a
    finding requires the governing argument to be attacker-influenced, never a
    compile-time constant."""
    if instr is None:
        return ""
    hlil = _safe(lambda: func.hlil)
    param_ids = set()
    for p in (_safe(lambda: list(func.parameter_vars)) or []):
        pid = _safe(lambda p=p: p.identifier)
        if pid is not None:
            param_ids.add(pid)
    tainted = {t.lower() for t in (tainted_source_set or set())}

    lines: List[str] = []
    call_exprs = list(_iter_call_exprs(instr)) or [instr]
    for ce in call_exprs:
        params = _safe(lambda ce=ce: list(ce.params)) or []
        for idx, p in enumerate(params):
            ptext = _safe(lambda p=p: str(p), "?")
            vars_ = _safe(lambda p=p: list(p.vars)) or []
            if not vars_:
                lines.append(f"- arg{idx} ({ptext}): COMPILE-TIME CONSTANT (not attacker-controlled)")
                continue
            tags: List[str] = []
            for v in vars_:
                vid = _safe(lambda v=v: v.identifier)
                if vid in param_ids:
                    tags.append("function parameter (tainted iff a caller passes tainted data)")
                if hlil is not None:
                    defs = _safe(lambda v=v: list(hlil.get_var_definitions(v))) or []
                    def_text = " ; ".join(_safe(lambda d=d: str(d), "") for d in defs)
                    low = def_text.lower()
                    hit = next((src for src in tainted if src and src in low), None)
                    if hit:
                        tags.append(f"derives from tainted source ({hit})")
                    elif "(" in def_text and ")" in def_text:
                        tags.append("assigned from a function call")
            if not tags:
                tags.append("local/computed value")
            seen = set()
            uniq = [t for t in tags if not (t in seen or seen.add(t))]
            lines.append(f"- arg{idx} ({ptext}): {', '.join(uniq)}")
    if not lines:
        return ""
    return ("Provenance of each call argument. A buffer-overflow or injection "
            "finding REQUIRES the governing argument (length/size/source/format/"
            "command/query/path) to be attacker-influenced; a COMPILE-TIME CONSTANT "
            "argument is not attacker-controlled and on its own is not a finding:\n"
            + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Binary Ninja value-set / constant analysis
# --------------------------------------------------------------------------- #
# Binary Ninja already runs constant propagation and value-set analysis over the
# IL. We mine that here so that sizes/lengths/indices that the decompiler has
# proven constant or bounded become GROUND TRUTH for the model and the offline
# heuristics, instead of being guessed. ``RegisterValueType`` member names are
# matched by substring so this works across Binary Ninja versions without
# importing the enum.

def expr_constant_value(expr):
    """The single concrete integer Binary Ninja proved for ``expr``, or None.

    Covers both literal constants and variables that constant-propagation resolved
    to a fixed value (e.g. ``len = 0x40`` upstream)."""
    rv = _safe(lambda: expr.value)
    if rv is None:
        return None
    tname = _safe(lambda: rv.type.name) or ""
    val = _safe(lambda: rv.value)
    if "Constant" in tname and isinstance(val, int):
        return val
    return None


def expr_max_value(expr):
    """An upper bound Binary Ninja proved for ``expr`` (a constant, or the largest
    end of a resolved value range), or None when undetermined. Used as a 'this
    value is constrained' signal — not, on its own, a safety proof."""
    c = expr_constant_value(expr)
    if c is not None:
        return c
    pvs = _safe(lambda: expr.possible_values)
    if pvs is None:
        return None
    tname = _safe(lambda: pvs.type.name) or ""
    if "Undetermined" in tname:
        return None
    ends = []
    for r in (_safe(lambda: list(pvs.ranges)) or []):
        end = _safe(lambda r=r: r.end)
        if isinstance(end, int):
            ends.append(end)
    for v in (_safe(lambda: list(pvs.values)) or []):
        if isinstance(v, int):
            ends.append(v)
    return max(ends) if ends else None


def _value_text(expr) -> str:
    """Short human description of the value/range BN resolved for ``expr`` ('' if
    nothing concrete is known)."""
    c = expr_constant_value(expr)
    if c is not None:
        return f"constant {c:#x} ({c})"
    pvs = _safe(lambda: expr.possible_values)
    if pvs is None:
        return ""
    tname = _safe(lambda: pvs.type.name) or ""
    if "Undetermined" in tname or not tname:
        return ""
    parts = []
    for r in (_safe(lambda: list(pvs.ranges)) or [])[:4]:
        start = _safe(lambda r=r: r.start)
        end = _safe(lambda r=r: r.end)
        if isinstance(start, int) and isinstance(end, int):
            parts.append(f"[{start:#x}, {end:#x}]")
    if parts:
        return "bounded range " + ", ".join(parts)
    vals = [v for v in (_safe(lambda: list(pvs.values)) or []) if isinstance(v, int)]
    if vals and len(vals) <= 8:
        return "one of {" + ", ".join(f"{v:#x}" for v in sorted(vals)) + "}"
    return ""


def argument_value_analysis(instr) -> str:
    """Per-argument concrete value / range from Binary Ninja's value-set analysis.

    This is the strongest deterministic signal for overflow reasoning: a length or
    size with a known constant or a bounded range can be compared directly against
    the destination capacity to decide the bug, rather than assumed."""
    if instr is None:
        return ""
    lines: List[str] = []
    for ce in (list(_iter_call_exprs(instr)) or [instr]):
        params = _safe(lambda ce=ce: list(ce.params)) or []
        for idx, p in enumerate(params):
            txt = _value_text(p)
            if txt:
                ptext = _safe(lambda p=p: str(p), "?")
                lines.append(f"- arg{idx} ({ptext}): {txt}")
    if not lines:
        return ""
    return ("Concrete values Binary Ninja's constant-propagation / value-set analysis "
            "resolved for the call arguments. Treat these as GROUND TRUTH: a length, "
            "size or index shown here as a constant or a bounded range is NOT freely "
            "attacker-controlled, and can be compared directly against the destination "
            "buffer capacity to decide whether an overflow is possible:\n"
            + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Stack frame layout
# --------------------------------------------------------------------------- #

def stack_layout_text(func, max_vars: int = 60) -> str:
    """Stack-frame variable layout (offsets + sizes) from Binary Ninja.

    For stack buffer overflows this grounds the reasoning: it shows each local's
    true byte size and where fixed-size arrays sit relative to the adjacent
    variables / saved frame pointer / return address, so a write past an array can
    be judged against what it would corrupt."""
    layout = _safe(lambda: list(func.stack_layout))
    if not layout:
        return ""
    rows = []
    arrays = 0
    for v in layout:
        name = _safe(lambda v=v: v.name) or "?"
        off = _safe(lambda v=v: v.storage)
        if not isinstance(off, int):
            continue
        t = _safe(lambda v=v: v.type)
        tstr = _safe(lambda t=t: str(t)) or "?"
        width = _safe(lambda t=t: t.width)
        count = _safe(lambda t=t: t.count)
        is_array = _safe(lambda t=t: t.element_type) is not None and isinstance(count, int)
        if is_array:
            arrays += 1
        wtxt = f", {width} byte(s)" if isinstance(width, int) else ""
        flag = "   <-- fixed-size array/buffer" if is_array else ""
        rows.append((off, f"  offset {off:#x}: {tstr} {name}{wtxt}{flag}"))
    # Only emit the section when there is at least one stack buffer to reason about;
    # otherwise it is noise.
    if not rows or arrays == 0:
        return ""
    rows.sort(key=lambda r: r[0])
    body = "\n".join(r[1] for r in rows[:max_vars])
    return ("Stack frame layout of the calling function (variable storage offsets and "
            "byte sizes from Binary Ninja; higher offsets are closer to the saved frame "
            "pointer and return address). A write past a fixed-size array can corrupt the "
            "variables at adjacent offsets and, if large enough, the saved return address "
            "— compare each array's capacity against the maximum bytes written to it:\n"
            + body)


def guard_conditions(instr, max_levels: int = 200) -> List[str]:
    """Path constraints: enclosing if/loop/switch conditions guarding ``instr``."""
    guards: List[str] = []
    cur = instr
    seen = 0
    while cur is not None and seen < max_levels:
        seen += 1
        parent = _safe(lambda c=cur: c.parent)
        if parent is None:
            break
        opname = _op_name(parent)
        cond = _safe(lambda p=parent: str(p.condition))
        if opname == "HLIL_IF":
            branch = "?"
            if _same_instr(_safe(lambda p=parent: p.true), cur):
                branch = "true"
            elif _same_instr(_safe(lambda p=parent: p.false), cur):
                branch = "false"
            guards.append(f"if ({cond})  -> reached when condition is {branch}")
        elif opname in ("HLIL_WHILE", "HLIL_WHILE_SSA"):
            guards.append(f"inside while ({cond})")
        elif opname in ("HLIL_DO_WHILE", "HLIL_DO_WHILE_SSA"):
            guards.append(f"inside do/while ({cond})")
        elif opname in ("HLIL_FOR", "HLIL_FOR_SSA"):
            guards.append(f"inside for (...; {cond}; ...)")
        elif opname in ("HLIL_SWITCH",):
            guards.append(f"inside switch ({cond})")
        cur = parent
    guards.reverse()
    return guards


def call_paths(bv, func, max_depth: int, max_paths: int) -> List[List]:
    """Backward paths from roots down to ``func`` (each a list of functions)."""
    results: List[List] = []

    def dfs(node, path, depth):
        if len(results) >= max_paths:
            return
        callers = _get_callers(bv, node)
        path_starts = {_safe(lambda p=p: p.start) for p in path}
        callers = [c for c in callers if _safe(lambda c=c: c.start) not in path_starts]
        if not callers or depth >= max_depth:
            results.append(list(reversed(path)))
            return
        for c in callers:
            if len(results) >= max_paths:
                return
            dfs(c, path + [c], depth + 1)

    dfs(func, [func], 0)
    return results[:max_paths]


def render_call_paths(bv, paths: List[List]) -> str:
    if not paths:
        return ("(no caller information available; function may be an entry "
                "point, exported, or only reached indirectly)")
    entry_starts = _entry_function_starts(bv)
    lines = []
    for path in paths:
        chain = " -> ".join(_safe(lambda f=f: f.name, "?") or "?" for f in path)
        root = path[0] if path else None
        flag = ""
        if root is not None:
            rs = _safe(lambda: root.start)
            if rs in entry_starts:
                flag = "  [root is an entry point / external surface]"
            elif not _get_callers(bv, root):
                flag = "  [root has no callers — top-level / exported / indirect]"
        lines.append(chain + flag)
    return "\n".join(lines)


def call_sites_between(bv, parent, child) -> List[int]:
    """Addresses inside ``parent`` that call ``child`` (best-effort)."""
    addrs: List[int] = []
    cstart = _safe(lambda: child.start)
    pstart = _safe(lambda: parent.start)
    if cstart is None or pstart is None:
        return addrs
    refs = _safe(lambda: list(bv.get_code_refs(cstart))) or []
    for r in refs:
        rf = _safe(lambda r=r: r.function)
        raddr = _safe(lambda r=r: r.address)
        if rf is not None and raddr is not None and _safe(lambda rf=rf: rf.start) == pstart:
            addrs.append(raddr)
    return addrs


def _ordered_path_ancestors(paths: List[List], caller_start) -> tuple:
    """From all paths, return (ordered ancestor funcs nearest-caller-first,
    mapping ancestor.start -> list of next-hop funcs toward the dangerous call)."""
    order: List = []
    seen = set()
    next_hops: dict = {}
    for path in paths:
        # path is root -> ... -> caller; walk from the caller backwards so the
        # functions closest to the dangerous call come first.
        for i in range(len(path) - 2, -1, -1):
            anc = path[i]
            child = path[i + 1]
            astart = _safe(lambda a=anc: a.start)
            if astart is None or astart == caller_start:
                continue
            hops = next_hops.setdefault(astart, [])
            cstart = _safe(lambda c=child: c.start)
            if cstart is not None and cstart not in {_safe(lambda h=h: h.start) for h in hops}:
                hops.append(child)
            if astart not in seen:
                seen.add(astart)
                order.append(anc)
    return order, next_hops


def collect_path_sibling_functions(bv, path_functions, exclude_starts, max_funcs: int):
    """Functions called BY the functions on the call path, other than the path
    itself — e.g. for MAIN -> ABCD -> strcpy, the functions MAIN and ABCD call
    elsewhere (XYZ, EFG). These may hold the bounds/validation checks that gate
    the dangerous value. Returns a list of ``(callee, via_function_name)``."""
    out: List = []
    seen = set(exclude_starts or set())
    for pf in path_functions:
        pf_name = _safe(lambda p=pf: p.name) or "?"
        for callee in (_safe(lambda p=pf: list(p.callees)) or []):
            if not _has_body(callee):
                continue
            cstart = _safe(lambda c=callee: c.start)
            if cstart is None or cstart in seen:
                continue
            seen.add(cstart)
            out.append((callee, pf_name))
            if len(out) >= max_funcs:
                return out
    return out


def render_sibling_function_section(bv, callee, via_name: str) -> str:
    name = _safe(lambda: callee.name) or "?"
    start = _safe(lambda: callee.start) or 0
    body, _rep = decompile_text(callee)
    return (f"function: {name} @ {start:#x}  (also called by {via_name} on the call "
            f"path; may contain a bounds/validation check)\n\n{body}")


def render_path_function_section(bv, ancestor, next_funcs: List) -> str:
    """Header (call sites + guards toward the next hop) + decompiled body."""
    name = _safe(lambda: ancestor.name) or "?"
    start = _safe(lambda: ancestor.start) or 0
    lines = [f"function: {name} @ {start:#x}  (on the path to the dangerous call)"]
    for nxt in next_funcs:
        nxt_name = _safe(lambda n=nxt: n.name) or "?"
        sites = call_sites_between(bv, ancestor, nxt)
        if not sites:
            lines.append(f"  calls {nxt_name} (call site not resolved)")
            continue
        for addr in sites:
            instr = hlil_instr_at(ancestor, addr)
            guards = guard_conditions(instr) if instr is not None else []
            if guards:
                guard_txt = "; ".join(guards)
                lines.append(f"  calls {nxt_name} at {addr:#x} when: {guard_txt}")
            else:
                lines.append(f"  calls {nxt_name} at {addr:#x} (unconditional in this function)")
    body, _rep = decompile_text(ancestor)
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def tainted_calls_in(func, source_set: set) -> List[str]:
    if not source_set:
        return []
    found = []
    callees = _safe(lambda: list(func.callees)) or []
    for c in callees:
        for nm in function_names(c):
            base = nm.split("@")[0]
            if base in source_set or nm in source_set:
                if base not in found:
                    found.append(base)
                break
    return found


# --------------------------------------------------------------------------- #
# String index (for phase 2 matching + context)
# --------------------------------------------------------------------------- #

def build_string_index(bv, max_per_func: int = 60) -> Dict[int, List[str]]:
    """Map function.start -> list of string values referenced within it."""
    index: Dict[int, List[str]] = {}
    strings = _safe(lambda: bv.get_strings()) or []
    for s in strings:
        val = _safe(lambda s=s: s.value)
        start = _safe(lambda s=s: s.start)
        if not val or start is None:
            continue
        refs = _safe(lambda s=s: list(bv.get_code_refs(start))) or []
        for r in refs:
            fn = _safe(lambda r=r: r.function)
            if fn is None:
                continue
            fstart = _safe(lambda fn=fn: fn.start)
            if fstart is None:
                continue
            lst = index.setdefault(fstart, [])
            if len(lst) < max_per_func and val not in lst:
                lst.append(val)
    return index


# --------------------------------------------------------------------------- #
# Variable data-flow (back-tracing the dangerous call's arguments)
# --------------------------------------------------------------------------- #

_CALL_OPS = ("HLIL_CALL", "HLIL_CALL_SSA", "HLIL_TAILCALL", "HLIL_TAILCALL_SSA")
# HLIL operations that are genuine compile-time literals (a string-literal argument
# is HLIL_CONST_PTR). A var-less expression that is NOT one of these (e.g. a function
# call result) must NOT be treated as a constant.
_CONST_OPS = ("HLIL_CONST", "HLIL_CONST_PTR", "HLIL_FLOAT_CONST", "HLIL_IMPORT",
              "HLIL_EXTERN_PTR")


def _iter_call_exprs(expr, depth: int = 0):
    """Yield HLIL call sub-expressions inside ``expr`` (recursively)."""
    if expr is None or depth > 64:
        return
    if _op_name(expr) in _CALL_OPS:
        yield expr
    for op in (_safe(lambda: expr.operands) or []):
        if hasattr(op, "operation"):
            yield from _iter_call_exprs(op, depth + 1)
        elif isinstance(op, (list, tuple)):
            for x in op:
                if hasattr(x, "operation"):
                    yield from _iter_call_exprs(x, depth + 1)


def _resolve_callee_function(bv, call_expr):
    """Resolve a call expression's target to a Function (or None)."""
    dest = _safe(lambda: call_expr.dest)
    if dest is None:
        return None
    addr = _safe(lambda: dest.constant)
    if not isinstance(addr, int):
        val = _safe(lambda: dest.value)
        addr = _safe(lambda: val.value) if val is not None else None
    if not isinstance(addr, int):
        return None
    return _safe(lambda: bv.get_function_at(addr))


def _has_body(func) -> bool:
    if func is None:
        return False
    return _safe(lambda: func.hlil) is not None or _safe(lambda: func.mlil) is not None


def call_arg_vars(instr) -> List:
    """Variables that appear as arguments of the call(s) inside ``instr``."""
    out: List = []
    seen = set()
    if instr is None:
        return out

    def add(v):
        ident = _safe(lambda: v.identifier)
        key = ident if ident is not None else id(v)
        if key not in seen:
            seen.add(key)
            out.append(v)

    call_exprs = list(_iter_call_exprs(instr))
    targets = call_exprs if call_exprs else [instr]
    for ce in targets:
        params = _safe(lambda ce=ce: list(ce.params))
        exprs = params if params else [ce]
        for e in exprs:
            for v in (_safe(lambda e=e: list(e.vars)) or []):
                add(v)
    return out


def call_param_is_constant(instr, idx: int) -> bool:
    """True if argument ``idx`` of the call in ``instr`` is a compile-time constant
    (its HLIL expression references no variables) — e.g. a string literal. Used to
    recognise that a printf-style format argument is fixed and therefore cannot be an
    uncontrolled format string."""
    if instr is None or idx < 0:
        return False
    for ce in (list(_iter_call_exprs(instr)) or [instr]):
        params = _safe(lambda ce=ce: list(ce.params))
        if params:
            if idx >= len(params):
                return False
            return not (_safe(lambda: list(params[idx].vars)) or [])
    return False


def call_param_is_literal_const(instr, idx: int) -> bool:
    """True only if argument ``idx`` of the call is a genuine compile-time literal
    (string/number literal), not merely variable-free. Stricter than
    ``call_param_is_constant`` — a function-call result is var-free but NOT a
    constant, so this avoids wrongly judging ``f(get_fmt())`` as a constant."""
    if instr is None or idx < 0:
        return False
    for ce in (list(_iter_call_exprs(instr)) or [instr]):
        params = _safe(lambda ce=ce: list(ce.params))
        if params:
            if idx >= len(params):
                return False
            return _op_name(params[idx]) in _CONST_OPS
    return False


def resolve_parameters_across_callers(bv, func, instr, max_callers: int = 30) -> str:
    """For each call argument that is a PARAMETER of the calling function, report what
    that function's callers actually pass for it (constant literal vs not), resolved
    across cross-references. Resolves the interprocedural case where a parameter looks
    'attacker-controlled' locally but every caller supplies a constant (e.g. a literal
    format string). '' if there is nothing to resolve."""
    if instr is None or bv is None:
        return ""
    # parameter identifier -> (index, name) for the calling function
    pid_to_idx: Dict = {}
    for i, p in enumerate(_safe(lambda: list(func.parameter_vars)) or []):
        pid = _safe(lambda p=p: p.identifier)
        if pid is not None:
            pid_to_idx[pid] = (i, _safe(lambda p=p: p.name) or f"arg{i}")
    if not pid_to_idx:
        return ""
    # which of those parameters appear as arguments of the dangerous call?
    used: Dict[int, str] = {}
    for v in call_arg_vars(instr):
        vid = _safe(lambda v=v: v.identifier)
        if vid in pid_to_idx:
            i, name = pid_to_idx[vid]
            used[i] = name
    if not used:
        return ""

    # collect up to max_callers distinct call sites of this function
    fstart = _safe(lambda: func.start)
    refs = (_safe(lambda: list(bv.get_code_refs(fstart))) or []) if fstart is not None else []
    sites = []
    seen = set()
    for r in refs:
        cf = _safe(lambda r=r: r.function)
        addr = _safe(lambda r=r: r.address)
        if cf is None or addr is None or addr in seen:
            continue
        seen.add(addr)
        ci = hlil_instr_at(cf, addr)
        if ci is not None:
            sites.append((_safe(lambda cf=cf: cf.name) or "?", ci))
            if len(sites) >= max_callers:
                break

    lines: List[str] = []
    for idx in sorted(used):
        name = used[idx]
        if not sites:
            lines.append(f"- parameter '{name}' (#{idx}): NO callers resolved in this binary "
                         "(may be exported or called only indirectly) — treat as POTENTIALLY "
                         "attacker-influenced.")
            continue
        n_const = sum(1 for _c, ci in sites if call_param_is_literal_const(ci, idx))
        n = len(sites)
        n_var = n - n_const
        if n_var == 0:
            lines.append(f"- parameter '{name}' (#{idx}): ALL {n} caller(s) pass a COMPILE-TIME "
                         "CONSTANT (e.g. a literal) — effectively constant on every resolved "
                         "path; NOT attacker-controlled via this parameter.")
        else:
            examples = [c for c, ci in sites if not call_param_is_literal_const(ci, idx)][:3]
            ex = f" (e.g. {', '.join(examples)})" if examples else ""
            lines.append(f"- parameter '{name}' (#{idx}): {n_var} of {n} caller(s) pass a "
                         f"NON-constant value{ex} — may be attacker-influenced on those paths; "
                         "evaluate them.")
    if not lines:
        return ""
    return ("These call arguments are PARAMETERS of the calling function. Here is what its "
            "callers actually pass for them, resolved across cross-references — use this "
            "instead of assuming a parameter is attacker-controlled. A parameter every "
            "caller sets to a constant (e.g. a literal format string) is NOT "
            "attacker-controlled:\n" + "\n".join(lines))


def dataflow_related_functions(bv, func, call_instr, arg_vars, max_funcs: int,
                               exclude_starts: Optional[set] = None):
    """Functions that produce/consume the call's argument variables.

    Returns ``(related, summary_text)`` where ``related`` is a list of
    ``(callee, var_name, relation)`` and ``summary_text`` describes where each
    argument variable is defined and used.
    """
    related: List = []
    summary_lines: List[str] = []
    hlil = _safe(lambda: func.hlil)
    if not arg_vars or hlil is None:
        return related, ""
    seen = set(exclude_starts or set())

    for var in arg_vars:
        vname = _safe(lambda v=var: v.name) or "?"
        defs = _safe(lambda v=var: list(hlil.get_var_definitions(v))) or []
        uses = _safe(lambda v=var: list(hlil.get_var_uses(v))) or []
        if defs or uses:
            summary_lines.append(f"argument variable '{vname}':")
        for d in defs[:6]:
            daddr = _safe(lambda d=d: d.address) or 0
            summary_lines.append(f"  assigned at {daddr:#x}: {_safe(lambda d=d: str(d), '')}")
            for ce in _iter_call_exprs(d):
                callee = _resolve_callee_function(bv, ce)
                if _has_body(callee):
                    cs = _safe(lambda c=callee: c.start)
                    if cs is not None and cs not in seen:
                        seen.add(cs)
                        related.append((callee, vname, "produces"))
        for u in uses[:8]:
            calls = list(_iter_call_exprs(u))
            if calls:
                uaddr = _safe(lambda u=u: u.address) or 0
                summary_lines.append(f"  used at {uaddr:#x}: {_safe(lambda u=u: str(u), '')}")
            for ce in calls:
                callee = _resolve_callee_function(bv, ce)
                if _has_body(callee):
                    cs = _safe(lambda c=callee: c.start)
                    if cs is not None and cs not in seen:
                        seen.add(cs)
                        related.append((callee, vname, "consumes"))
        if len(related) >= max_funcs:
            break

    return related[:max_funcs], "\n".join(summary_lines)


def render_related_function_section(bv, callee, var_name: str, relation: str) -> str:
    name = _safe(lambda: callee.name) or "?"
    start = _safe(lambda: callee.start) or 0
    body, _rep = decompile_text(callee)
    return (f"function: {name} @ {start:#x}  ({relation} the call argument "
            f"'{var_name}')\n\n{body}")


# --------------------------------------------------------------------------- #
# Data types used across the call chain
# --------------------------------------------------------------------------- #

def _collect_named_type_refs(t, acc: set, depth: int = 0):
    if t is None or depth > 6:
        return
    ntr = _safe(lambda: t.named_type_reference)
    if ntr is not None:
        nm = _safe(lambda: ntr.name)
        if nm is not None:
            acc.add(str(nm))
    for attr in ("target", "element_type", "child_type"):
        sub = _safe(lambda a=attr: getattr(t, a, None))
        if sub is not None and sub is not t:
            _collect_named_type_refs(sub, acc, depth + 1)


def _function_type_names(func, acc: set):
    ft = _safe(lambda: func.type)
    if ft is not None:
        _collect_named_type_refs(ft, acc)
        for p in (_safe(lambda: list(func.type.parameters)) or []):
            _collect_named_type_refs(_safe(lambda p=p: p.type), acc)
    for v in (_safe(lambda: list(func.vars)) or []):
        _collect_named_type_refs(_safe(lambda v=v: v.type), acc)


def _struct_members(t):
    m = _safe(lambda: list(t.members))
    if m:
        return m
    st = _safe(lambda: t.structure)
    if st is not None:
        return _safe(lambda: list(st.members)) or []
    return []


def _render_type_definition(bv, name: str) -> Optional[str]:
    t = _safe(lambda: bv.get_type_by_name(name))
    if t is None:
        return None
    members = _struct_members(t)
    if members:
        total = _safe(lambda: t.width)
        head = f"struct {name} {{" + (f"   // total {total:#x} bytes" if isinstance(total, int) else "")
        lines = [head]
        for m in members:
            mt = _safe(lambda m=m: str(m.type)) or "?"
            mn = _safe(lambda m=m: m.name) or "?"
            moff = _safe(lambda m=m: m.offset)
            mw = _safe(lambda m=m: m.type.width)
            ann = []
            if isinstance(moff, int):
                ann.append(f"offset {moff:#x}")
            if isinstance(mw, int):
                ann.append(f"{mw} byte(s)")
            suffix = f"   // {', '.join(ann)}" if ann else ""
            lines.append(f"    {mt} {mn};{suffix}")
        lines.append("};")
        return "\n".join(lines)
    enum = _safe(lambda: t.enumeration)
    if enum is not None:
        ems = _safe(lambda: list(enum.members)) or []
        body = ", ".join(
            f"{_safe(lambda e=e: e.name)}={_safe(lambda e=e: e.value)}" for e in ems
        )
        return f"enum {name} {{ {body} }};"
    rendered = _safe(lambda: str(t))
    return f"typedef {rendered} {name};" if rendered else None


def collect_type_definitions(bv, funcs, max_types: int) -> str:
    acc: set = set()
    for f in funcs:
        _function_type_names(f, acc)
    if not acc:
        return ""
    blocks = []
    for name in sorted(acc)[:max_types]:
        d = _render_type_definition(bv, name)
        if d:
            blocks.append(d)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #

def _binary_header(bv, func, target_name: str, call_addr: int, representation: str) -> str:
    bin_name = _safe(lambda: bv.file.original_filename) or _safe(lambda: bv.file.filename) or "(unknown)"
    arch = _safe(lambda: bv.arch.name) or "?"
    platform = _safe(lambda: bv.platform.name) or "?"
    fname = _safe(lambda: func.name) or "?"
    fstart = _safe(lambda: func.start) or 0
    return (
        f"binary: {bin_name}\n"
        f"arch/platform: {arch} / {platform}\n"
        f"calling function: {fname} @ {fstart:#x}\n"
        f"dangerous call target: {target_name}\n"
        f"call site address: {call_addr:#x}\n"
        f"context representation: {representation}"
    )


def callee_prototype_text(bv, target_name: str) -> str:
    """Declared C prototype of the called function, '' if unknown.

    The curated table is authoritative for the dangerous functions whose argument
    layout trips up position-based reasoning (fortified ``__*_chk`` and bounds-checked
    ``*_s`` variants), so it is preferred. For anything not in the table we fall back
    to Binary Ninja's own function type when it is a real signature.

    Note: do NOT use ``Symbol.type`` here — that is a ``SymbolType`` *enum*
    (e.g. ImportedFunctionSymbol), not a prototype.
    """
    proto = _prototypes.lookup_prototype(target_name)
    if proto:
        return proto
    for f in (_safe(lambda: bv.get_functions_by_name(target_name)) or []):
        s = _safe(lambda f=f: str(f.type)) if f is not None else None
        if s and "(" in s:  # looks like a function signature, not a bare type
            return s.strip()
    return ""


def build_call_site_context(bv, func, target_name: str, call_addr: int, cfg,
                            tainted_source_set: Optional[set] = None) -> ContextBundle:
    """Phase 1 context for a dangerous call site, budgeted to cfg.max_context_tokens."""
    body_text, representation = decompile_text(func)
    instr = hlil_instr_at(func, call_addr)
    call_text = render_call(instr)
    guards = guard_conditions(instr) if instr is not None else []
    paths = call_paths(bv, func, cfg.call_path_max_depth, cfg.call_path_max_paths)
    paths_text = render_call_paths(bv, paths)
    taint = tainted_calls_in(func, tainted_source_set or set())

    caller_start = _safe(lambda: func.start)
    ancestors, next_hops = _ordered_path_ancestors(paths, caller_start)
    max_bodies = max(0, int(getattr(cfg, "call_path_max_bodies", 12)))
    include_bodies = getattr(cfg, "call_path_include_bodies", True)
    included_ancestors = ancestors[:max_bodies] if include_bodies else []

    # Argument variables of the dangerous call (used for sizes + data-flow).
    arg_vars = call_arg_vars(instr)
    # Prefer expression-type capacities (resolves struct fields / pointed-to
    # arrays); fall back to the declared variable sizes.
    sizes_text = arg_buffer_sizes_text(instr) or arg_var_sizes_text(arg_vars)
    provenance_text = argument_provenance(func, instr, tainted_source_set or set())
    value_text = argument_value_analysis(instr)
    param_resolution_text = resolve_parameters_across_callers(bv, func, instr)
    stack_text = stack_layout_text(func) if getattr(cfg, "include_stack_layout", True) else ""

    # Variable data-flow: which functions produce/consume the call's arguments.
    dataflow_funcs: List = []
    dataflow_summary = ""
    if getattr(cfg, "include_variable_dataflow", True):
        exclude = {caller_start} | {_safe(lambda a=a: a.start) for a in included_ancestors}
        dataflow_funcs, dataflow_summary = dataflow_related_functions(
            bv, func, instr, arg_vars,
            max_funcs=max(0, int(getattr(cfg, "dataflow_max_functions", 8))),
            exclude_starts=exclude,
        )

    # Sibling/callee functions of the path functions (may hold bounds checks that
    # gate the dangerous value). Filled only while the budget allows.
    sibling_funcs: List = []
    if getattr(cfg, "include_call_path_siblings", True):
        path_functions = [func] + included_ancestors
        sib_exclude = ({caller_start}
                       | {_safe(lambda a=a: a.start) for a in included_ancestors}
                       | {_safe(lambda c=c: c.start) for c, _v, _r in dataflow_funcs})
        sibling_funcs = collect_path_sibling_functions(
            bv, path_functions, sib_exclude,
            max_funcs=max(0, int(getattr(cfg, "call_path_sibling_max_bodies", 12))),
        )

    # Data types referenced across the whole call chain + argument variables.
    types_text = ""
    if getattr(cfg, "include_data_types", True):
        type_funcs = ([func] + included_ancestors
                      + [c for c, _v, _r in dataflow_funcs]
                      + [c for c, _v in sibling_funcs])
        types_text = collect_type_definitions(
            bv, type_funcs, max_types=max(0, int(getattr(cfg, "max_type_defs", 24)))
        )

    budget = _tokens.Budget(cfg.max_context_tokens, cfg.tokenizer_encoding)
    budget.add("OVERVIEW", _binary_header(bv, func, target_name, call_addr, representation),
               truncatable=False)
    budget.add("DANGEROUS CALL (and arguments)", call_text, truncatable=False)
    proto = callee_prototype_text(bv, target_name)
    if proto:
        budget.add(
            "DECLARED PROTOTYPE OF THE CALLED FUNCTION",
            f"{proto}\n"
            "Map the call's arguments to these parameters before judging. Fortified "
            "(__*_chk) and bounds-checked (*_s) variants take EXTRA leading arguments "
            "(e.g. a flag and/or size), so the format string, size, or destination is "
            "at a SHIFTED position versus the base function. A constant string-literal "
            "format argument is a controlled format, not an uncontrolled-format bug.",
            truncatable=False)
    if sizes_text:
        budget.add("ARGUMENT VARIABLE TYPES AND SIZES (buffer capacities)",
                   sizes_text, truncatable=False)
    if value_text:
        budget.add("ARGUMENT VALUE ANALYSIS (constants/ranges Binary Ninja resolved)",
                   value_text, truncatable=False)
    if provenance_text:
        budget.add("ARGUMENT PROVENANCE (is each argument attacker-influenced?)",
                   provenance_text, truncatable=False)
    if param_resolution_text:
        budget.add("PARAMETER RESOLUTION (what callers pass for this function's parameters)",
                   param_resolution_text, truncatable=False)
    if guards:
        budget.add("PATH CONSTRAINTS (guards reaching the call)",
                   "\n".join(f"- {g}" for g in guards), truncatable=False)
    else:
        budget.add("PATH CONSTRAINTS (guards reaching the call)",
                   "- none (call is unconditional within this function)",
                   truncatable=False)
    if dataflow_summary:
        budget.add("ARGUMENT DATA-FLOW (where each call argument is defined/used)",
                   dataflow_summary, truncatable=True)
    if taint:
        budget.add("POSSIBLE TAINTED-INPUT SOURCES CALLED IN THIS FUNCTION",
                   "These input sources are called within the function and may "
                   "feed the dangerous call: " + ", ".join(taint), truncatable=False)
    if stack_text:
        budget.add("STACK FRAME LAYOUT OF THE CALLING FUNCTION",
                   stack_text, truncatable=False)
    budget.add("DECOMPILED BODY OF THE CALLING FUNCTION", body_text, truncatable=True)
    if types_text:
        budget.add("DATA TYPES USED IN THE CALL CHAIN", types_text, truncatable=True)

    # Bodies of functions that produce/consume the dangerous call's arguments.
    for callee, vname, relation in dataflow_funcs:
        cname = _safe(lambda c=callee: c.name) or "?"
        budget.add(
            f"DATA-FLOW FUNCTION BODY: {cname} ({relation} arg '{vname}')",
            render_related_function_section(bv, callee, vname, relation),
            truncatable=True,
        )

    budget.add("CALL PATHS TO THIS FUNCTION (summary)", paths_text, truncatable=True)

    # Bodies of every function along the call path(s), nearest the dangerous call
    # first, each annotated with the call site + guards leading to the next hop.
    # This lets the model evaluate the conditions that gate reaching the call.
    for anc in included_ancestors:
        astart = _safe(lambda a=anc: a.start)
        section = render_path_function_section(bv, anc, next_hops.get(astart, []))
        aname = _safe(lambda a=anc: a.name) or "?"
        budget.add(f"CALL-PATH FUNCTION BODY: {aname}", section, truncatable=True)

    # Lowest priority: bodies of other functions called by the path functions —
    # they may contain the bounds/validation checks that prevent the bug.
    for callee, via_name in sibling_funcs:
        cname = _safe(lambda c=callee: c.name) or "?"
        budget.add(
            f"RELATED FUNCTION CALLED ON THE PATH (via {via_name}): {cname}",
            render_sibling_function_section(bv, callee, via_name),
            truncatable=True,
        )

    return ContextBundle(
        text=budget.render(),
        representation=representation,
        call_paths_text=paths_text,
        tainted_sources=taint,
        truncated=budget.truncated_labels,
    )


def build_function_context(bv, func, cfg, referenced_strings: Optional[Sequence[str]] = None,
                           matched_apis: Optional[Sequence[str]] = None) -> ContextBundle:
    """Phase 2 context for a security-sensitive function."""
    body_text, representation = decompile_text(func)
    paths = call_paths(bv, func, cfg.call_path_max_depth, cfg.call_path_max_paths)
    paths_text = render_call_paths(bv, paths)

    fname = _safe(lambda: func.name) or "?"
    fstart = _safe(lambda: func.start) or 0
    bin_name = _safe(lambda: bv.file.original_filename) or "(unknown)"
    header = (
        f"binary: {bin_name}\n"
        f"function: {fname} @ {fstart:#x}\n"
        f"context representation: {representation}"
    )

    types_text = ""
    if getattr(cfg, "include_data_types", True):
        types_text = collect_type_definitions(
            bv, [func], max_types=max(0, int(getattr(cfg, "max_type_defs", 24)))
        )

    budget = _tokens.Budget(cfg.max_context_tokens, cfg.tokenizer_encoding)
    budget.add("OVERVIEW", header, truncatable=False)
    if matched_apis:
        budget.add("SECURITY-RELEVANT APIs CALLED",
                   ", ".join(sorted(set(matched_apis))), truncatable=False)
    if referenced_strings:
        joined = "\n".join(f"- {s!r}" for s in list(referenced_strings)[:60])
        budget.add("STRINGS REFERENCED IN THIS FUNCTION", joined, truncatable=True)
    budget.add("DECOMPILED BODY", body_text, truncatable=True)
    if types_text:
        budget.add("DATA TYPES USED IN THIS FUNCTION", types_text, truncatable=True)
    budget.add("CALL PATHS TO THIS FUNCTION", paths_text, truncatable=True)

    return ContextBundle(
        text=budget.render(),
        representation=representation,
        call_paths_text=paths_text,
        truncated=budget.truncated_labels,
    )


def security_apis_in(func, name_substrings: Sequence[str]) -> List[str]:
    """Callee names whose name contains any of ``name_substrings`` (lowercased)."""
    subs = [s.lower() for s in name_substrings]
    found = []
    callees = _safe(lambda: list(func.callees)) or []
    for c in callees:
        for nm in function_names(c):
            low = nm.lower()
            if any(s in low for s in subs):
                if nm not in found:
                    found.append(nm)
                break
    return found
