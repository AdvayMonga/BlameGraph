"""Reconstruct a run's experiment log: config versions, server restarts, eval launches, observed metrics."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .traces import Run, Step

SERVER_FILE = "start_server.sh"
STUB_MARK = "has no engine configured"

# ---- regexes over shell commands ----
# a real launch: `python3 evaluate.py`, `./evaluate.py`, `python -u /home/user/task/evaluate.py`
EVAL_LAUNCH_RE = re.compile(r"(?:python3?(?:\s+-\w+)*\s+\S*|\./)evaluate\.py\b")
EVAL_KILL_RE = re.compile(r"\b(pkill|kill|killall)\b[^\n|;&]*evaluate\.py")
QUICK_RE = re.compile(r"--quick\b")
# a "standard" eval uses only the harness's own flags and no INFERENCE_BENCH_* overrides (except the base model)
STANDARD_FLAGS = {"--quick", "--json-output-file", "--server-url", "--model", "--host", "--port"}
NONSTD_ENV_RE = re.compile(r"\bINFERENCE_BENCH_(?!BASE_MODEL\b)\w+\s*=")
# in-place edits of the grader from the shell: sed -i, patch, tee/redirect into it, cp/mv over it
EVAL_TAMPER_RE = re.compile(
    r"(sed\s+-i[^\n|;&]*evaluate\.py|(?:^|[|;&]\s*)patch\b[^\n|;&]*evaluate\.py|apply_patch[^\n]*evaluate\.py|"
    r"(?:>|tee)\s*\S*/?evaluate\.py\b|\b(?:cp|mv)\s+\S+\s+\S*evaluate\.py\b)")
SERVER_START_RE = re.compile(
    r"(start_server\.sh|test_server\.sh|vllm\s+serve|vllm\.entrypoints|sglang\.launch_server|"
    r"sglang\s+serve|text-generation-launcher|lmdeploy\s+serve|trtllm-serve|uvicorn\s)"
)
TIMER_RE = re.compile(r"Remaining time \(hours:minutes\):\s*\n?\s*(\d+):(\d\d)")
READ_ONLY_RE = re.compile(r"^\s*(cat|sed\s+-n|head|tail|less|grep|bat|wc|ls|stat|diff|ps\b|pgrep)\b")
# metric sightings in tool output: JSON, python dict, or "ttft p50 0.12" prose
P50_RE = re.compile(
    r"[\"']?(ttft|tpot|itl)[\"']?\s*(?::|=|\s)\s*\{?\s*[\"']?p50[\"']?\s*(?::|=|\s)\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)",
    re.I,
)
P50_PROSE_RE = re.compile(r"\b(ttft|tpot)\b[^\n]{0,40}?p50[^\d\n]{0,12}([0-9]*\.?[0-9]+)\s*(ms|s)?\b", re.I)
# "ttft_p50: 2.25 ms", "tpot_p50 0.0097", "speed: ttft_p50 0.17"
P50_FLAT_RE = re.compile(r"\b(ttft|tpot)[_\s-]?p50\b[^0-9\n]{0,8}([0-9]*\.?[0-9]+)\s*(ms|s)?\b", re.I)
RPS_RE = re.compile(
    r"(?:request_throughput_req_per_s|req(?:uest)?[_\s]?(?:ps|/s|_per_s|TP|_throughput)|requests?/s(?:ec)?)[\"']?\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.I)
GEN_TPS_RE = re.compile(
    r"(?:generation_throughput_tokens_per_s|gen(?:eration)?[_\s]?(?:tps|TP|throughput)|tokens?/s(?:ec)?)[^0-9\n]{0,12}([0-9]*\.?[0-9]+)", re.I)
FAILRATE_RE = re.compile(r"fail(?:ure)?[_\s]?(?:rate)?[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
SERVER_LOG_LINE_RE = re.compile(r"\(APIServer pid=\d+\)|Avg prompt throughput|Avg generation throughput")
QUALITY_RE = re.compile(r"[\"']?pass[\"']?\s*:\s*(true|false)", re.I)
MMLU_RATIO_RE = re.compile(r"[\"']?ratio[\"']?\s*:\s*([0-9]*\.?[0-9]+)")
OBS_KW_RE = re.compile(r"p50|throughput|fail|req_ps|reqTP|tok(?:ens)?/s|tps", re.I)


@dataclass
class ConfigVersion:
    idx: int                  # 0-based version number
    step_i: int               # event index where it was written
    content: str
    minute: float | None      # elapsed minutes (from nearest earlier timer.sh), if known

    @property
    def hash(self) -> str:
        return hashlib.sha1(normalize(self.content).encode()).hexdigest()[:10]

    @property
    def engine(self) -> str:
        c = self.content.lower()
        for name, pat in (("vllm", r"vllm"), ("sglang", r"sglang"), ("tgi", r"text-generation|tgi"),
                          ("trtllm", r"trtllm|tensorrt"), ("lmdeploy", r"lmdeploy")):
            if re.search(pat, c):
                return name
        if STUB_MARK in c:
            return "stub"
        return "custom" if re.search(r"python|uvicorn|exec", c) else "unknown"

    @property
    def flags(self) -> dict[str, str]:
        """--flag value pairs in the launch command (value '' for boolean flags)."""
        out: dict[str, str] = {}
        body = re.sub(r"\\\n", " ", self.content)
        for m in re.finditer(r"(--[a-zA-Z][\w-]*)(?:[= ]((?:\"[^\"]*\"|'[^']*'|[^\s\"'\\-][^\s\"'\\]*)))?", body):
            out[m.group(1)] = (m.group(2) or "").strip("\"'")
        return out


@dataclass
class EvalLaunch:
    step_i: int
    quick: bool
    config_idx: int | None        # config version believed live on the server at launch
    config_stale: bool            # config file changed since the last server start
    minute: float | None
    error: bool
    observed: bool = False        # a metric observation followed before the next launch
    killed: bool = False          # agent explicitly killed evaluate.py before observing
    standard: bool = True         # harness-default invocation (comparable to the final eval)
    warm: bool = False            # an earlier eval already ran on this server instance (caches warm)


@dataclass
class Observation:
    step_i: int
    ttft_p50: float | None = None
    tpot_p50: float | None = None
    rps: float | None = None
    rps_geomean: float | None = None   # geomean over burst/poisson/constant when all three were reported (scenario C metric)
    gen_tps: float | None = None
    failure_rate: float | None = None
    quality_pass: bool | None = None
    mmlu_ratio: float | None = None
    config_idx: int | None = None   # config believed measured (live at the preceding eval launch)
    quick: bool | None = None       # whether the preceding eval launch was a --quick smoke test
    standard: bool | None = None    # whether the preceding eval launch was a standard invocation
    warm: bool | None = None        # whether the preceding eval ran on an already-exercised server instance


@dataclass
class ExperimentLog:
    run_id: str
    configs: list[ConfigVersion] = field(default_factory=list)
    evals: list[EvalLaunch] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    server_starts: list[tuple[int, int | None]] = field(default_factory=list)   # (step_i, config_idx launched)
    timer_marks: list[tuple[int, float]] = field(default_factory=list)   # (step_i, elapsed_min)
    eval_script_modified: bool = False
    eval_script_step: int | None = None   # first step that modified evaluate.py (traces only)
    n_eval_kills: int = 0

    # ---- derived ----
    @property
    def final_config(self) -> ConfigVersion | None:
        return self.configs[-1] if self.configs else None

    def _idx_to_hash(self) -> dict[int, str]:
        return {c.idx: c.hash for c in self.configs}

    @property
    def n_distinct_configs(self) -> int:
        return len({c.hash for c in self.configs if c.engine != "stub"})

    @property
    def n_measured_distinct(self) -> int:
        h = self._idx_to_hash()
        return len({h[e.config_idx] for e in self.evals if e.config_idx in h})

    @property
    def n_observed_distinct(self) -> int:
        h = self._idx_to_hash()
        return len({h[o.config_idx] for o in self.observations if o.config_idx in h})

    @property
    def final_config_measured(self) -> bool:
        """Was the final config (or an identical one) ever the live target of an eval launch?"""
        f = self.final_config
        if not f:
            return False
        h = self._idx_to_hash()
        return any(e.config_idx in h and h[e.config_idx] == f.hash for e in self.evals)

    @property
    def final_config_full_eval(self) -> bool:
        f = self.final_config
        if not f:
            return False
        h = self._idx_to_hash()
        return any(not e.quick and e.observed and e.config_idx in h and h[e.config_idx] == f.hash for e in self.evals)

    @property
    def stale_eval_count(self) -> int:
        return sum(1 for e in self.evals if e.config_stale)

    @property
    def unobserved_eval_count(self) -> int:
        return sum(1 for e in self.evals if not e.observed)

    def launched_configs(self) -> list[tuple[int, str]]:
        """(step_i, config_hash) each time a *different* config is launched, in order."""
        h = self._idx_to_hash(); out = []
        for si, idx in self.server_starts:
            if idx in h and (not out or out[-1][1] != h[idx]):
                out.append((si, h[idx]))
        return out

    def primary_observations(self, scenario: str) -> list[tuple[int, float]]:
        """(config_idx, higher-is-better metric) for observations that carry the scenario's metric."""
        out = []
        for o in self.observations:
            if o.config_idx is None:
                continue
            v = None
            if scenario == "A" and o.ttft_p50:
                v = 1 / o.ttft_p50
            elif scenario == "B" and o.tpot_p50:
                v = 1 / o.tpot_p50
            elif scenario in ("C", "D") and o.rps:
                v = o.rps  # approximation: D is a geomean, C averages 3 profiles
            if v is not None:
                out.append((o.config_idx, v))
        return out

    def regret(self, scenario: str) -> float | None:
        """best observed metric / metric observed for the final config (>1 means a better config was seen and dropped)."""
        obs = self.primary_observations(scenario)
        f = self.final_config
        if not obs or not f:
            return None
        h = self._idx_to_hash()
        final_vals = [v for c, v in obs if h.get(c) == f.hash]
        if not final_vals:
            return None
        return max(v for _, v in obs) / max(final_vals)


def normalize(s: str) -> str:
    """Whitespace/comment-insensitive form of a shell script for hashing."""
    lines = []
    for ln in s.splitlines():
        ln = re.sub(r"(?<!\\)#.*$", "", ln).strip()
        if ln:
            lines.append(re.sub(r"\s+", " ", ln))
    return "\n".join(lines)


def _apply_edit(content: str, old: str, new: str, replace_all: bool) -> str | None:
    if old == "":
        return new  # claude Edit with empty old_string == whole-file write
    if old not in content:
        return None
    return content.replace(old, new) if replace_all else content.replace(old, new, 1)


def _content_from_codex_diff(diff: str) -> str | None:
    """Reconstruct start_server.sh from codex's cumulative diff (context + added lines of its hunks)."""
    m = re.search(r"diff --git a/+\S*" + re.escape(SERVER_FILE) + r".*?(?=\ndiff --git |\Z)", diff, re.S)
    if not m:
        return None
    lines, in_hunk = [], False
    for ln in m.group(0).splitlines():
        if ln.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and ln[:1] in ("+", " "):
            lines.append(ln[1:])
    return "\n".join(lines) if lines else None


def _shell_write(cmd: str) -> str | None:
    """Content of start_server.sh written via a heredoc in a shell command, if detectable."""
    for pat in (r"cat\s*>\s*\S*" + SERVER_FILE + r"\s*<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n\1\b",
                r"cat\s*<<-?\s*['\"]?(\w+)['\"]?\s*>\s*\S*" + SERVER_FILE + r"\n(.*?)\n\1\b"):
        m = re.search(pat, cmd, re.S)
        if m:
            return m.group(2)
    return None


def _touches_server_file(cmd: str) -> bool:
    return SERVER_FILE in cmd and bool(re.search(
        r"(sed\s+-i|tee\b|>\s*\S*" + SERVER_FILE + r"|cp\s+\S+\s+\S*" + SERVER_FILE + r"|mv\s+\S+\s+\S*" + SERVER_FILE + r")", cmd))


SHELL_WRAP_RE = re.compile(r"""^/bin/bash -lc (['"])(.*)\1(?: in \S+)?$""", re.S)
SHELL_WRAP_OPEN_RE = re.compile(r"""^/bin/bash -lc ['"]""")
# codex traces keep only the first line of multi-line commands; outputs still show server launches
STARTUP_BANNER_RE = re.compile(
    r"Application startup complete|Uvicorn running on|=== Inference server ===|Started server process|"
    r"READY after|launched pid|server started|Server OK", re.I)


def unwrap_shell(cmd: str) -> str:
    """`/bin/bash -lc '<inner>' in /dir` (codex) -> `<inner>`; tolerates truncated (unclosed) commands."""
    m = SHELL_WRAP_RE.match(cmd.strip())
    if m:
        return m.group(2)
    return SHELL_WRAP_OPEN_RE.sub("", cmd.strip())


CODEX_PREFIX_RE = re.compile(r"^(succeeded|failed|exited \S+) in [^:\n]+:\n")
READ_LINENO_RE = re.compile(r"^\s*\d+→", re.M)
FULL_CAT_RE = re.compile(r"^\s*(cat|bat\s+-p|sed\s+-n\s+['\"]?1,(\d+)p['\"]?)\s+\S*" + SERVER_FILE + r"\s*$")


def _full_read_of_server_file(tool: str, cmd: str, path: str, ti: dict, out: str) -> str | None:
    """Content of start_server.sh if this step read the whole file; None otherwise."""
    if not out or "No such file" in out[:200]:
        return None
    if tool == "read" and path.endswith(SERVER_FILE) and not ti.get("offset") and not ti.get("limit"):
        body = READ_LINENO_RE.sub("", out)
        return body if "\n" in body else None
    if tool in ("bash", "shell"):
        m = FULL_CAT_RE.match(cmd.strip())
        if m and (m.group(2) is None or int(m.group(2)) >= 150):
            body = CODEX_PREFIX_RE.sub("", out)
            return body if body.count("\n") >= 3 else None
    return None


SEG_SPLIT_RE = re.compile(r"\|\||&&|[|;\n]")
NON_LAUNCH_HEAD_RE = re.compile(r"^\s*(grep|pgrep|ps|pkill|kill|echo|printf|cat|sed|head|tail|less|ls|chmod|wc|test|\[)\b")


def _is_standard_eval(cmd: str) -> bool:
    """True when every flag after evaluate.py is a harness flag and no INFERENCE_BENCH_* override is set."""
    if NONSTD_ENV_RE.search(cmd):
        return False
    for seg in SEG_SPLIT_RE.split(cmd):
        m = EVAL_LAUNCH_RE.search(seg)
        if not m or NON_LAUNCH_HEAD_RE.match(seg):
            continue
        flags = set(re.findall(r"(--[\w-]+)", seg[m.end():]))
        if flags - STANDARD_FLAGS:
            return False
    return True


def _launches_eval(cmd: str) -> bool:
    """True when some pipeline segment actually executes evaluate.py (not grep/ps/echo mentioning it)."""
    for seg in SEG_SPLIT_RE.split(cmd):
        if NON_LAUNCH_HEAD_RE.match(seg):
            continue
        if EVAL_LAUNCH_RE.search(seg):
            return True
    return False


def _extract_observation(out: str, step_i: int, config_idx: int | None) -> Observation | None:
    obs = Observation(step_i=step_i, config_idx=config_idx)
    for m in P50_RE.finditer(out):
        k, v = m.group(1).lower(), float(m.group(2))
        if k == "ttft" and obs.ttft_p50 is None:
            obs.ttft_p50 = v
        elif k == "tpot" and obs.tpot_p50 is None:
            obs.tpot_p50 = v
    if obs.ttft_p50 is None and obs.tpot_p50 is None:
        for rx in (P50_FLAT_RE, P50_PROSE_RE):
            for m in rx.finditer(out):
                k, v, unit = m.group(1).lower(), float(m.group(2)), (m.group(3) or "").lower()
                if unit == "ms" or (unit == "" and v > 30):   # bare numbers above 30 are milliseconds
                    v = v / 1000.0
                if v > 30 or v <= 0:
                    continue
                if k == "ttft" and obs.ttft_p50 is None:
                    obs.ttft_p50 = v
                elif k == "tpot" and obs.tpot_p50 is None:
                    obs.tpot_p50 = v
    clean = "\n".join(l for l in out.splitlines() if not SERVER_LOG_LINE_RE.search(l))
    if m := RPS_RE.search(clean):
        obs.rps = float(m.group(1))
    if m := GEN_TPS_RE.search(clean):
        obs.gen_tps = float(m.group(1))
    if m := FAILRATE_RE.search(clean):
        v = float(m.group(1))
        obs.failure_rate = v / 100.0 if v > 1 else v
    if (m := MMLU_RATIO_RE.search(out)) and "mmlu" in out.lower():
        obs.mmlu_ratio = float(m.group(1))
        if q := QUALITY_RE.search(out):
            obs.quality_pass = q.group(1).lower() == "true"
    if any(v is not None for v in (obs.ttft_p50, obs.tpot_p50, obs.rps, obs.gen_tps, obs.failure_rate)):
        return obs
    return None


def build_log(run: Run) -> ExperimentLog:
    """Reconstruct the experiment log from a raw trace (sessions recorded at the source use adapter.log_from_ledger)."""
    log = ExperimentLog(run_id=run.run_id)
    content: str | None = None
    live_idx: int | None = None        # config version running on the server
    evals_on_instance = 0              # eval launches since the last server start
    elapsed: float | None = None

    def add_version(step_i: int, new_content: str):
        nonlocal content
        if content is not None and normalize(content) == normalize(new_content):
            return
        content = new_content
        log.configs.append(ConfigVersion(idx=len(log.configs), step_i=step_i, content=new_content, minute=elapsed))

    def cur_idx() -> int | None:
        return log.configs[-1].idx if log.configs else None

    for st in run.steps():
        cmd, tool, out = unwrap_shell(st.cmd), st.tool, st.output
        ti = st.call.tool_input if isinstance(st.call.tool_input, dict) else {}
        path = str(ti.get("file_path") or ti.get("filePath") or "")
        is_shell = tool in ("bash", "shell")

        for m in TIMER_RE.finditer(out):
            elapsed = 120.0 - (int(m.group(1)) * 60 + int(m.group(2)))
            log.timer_marks.append((st.i, elapsed))

        # -- config versions written through file tools or heredocs
        if tool == "write" and path.endswith(SERVER_FILE):
            add_version(st.i, str(ti.get("content", "")))
        elif tool in ("edit", "multiedit") and path.endswith(SERVER_FILE):
            old = str(ti.get("old_string", ti.get("oldString", "")))
            new = str(ti.get("new_string", ti.get("newString", "")))
            ra = bool(ti.get("replace_all", ti.get("replaceAll", False)))
            upd = _apply_edit(content or "", old, new, ra)
            if upd is not None:
                add_version(st.i, upd)
            elif content is None:
                add_version(st.i, new)
        elif is_shell:
            hw = _shell_write(cmd)
            if hw is not None:
                add_version(st.i, hw)
            elif _touches_server_file(cmd) and not READ_ONLY_RE.match(cmd) and st.codex_diff is None:
                add_version(st.i, (content or "") + f"\n# <modified by: {cmd[:80]}>")
        # passive reconstruction: a full read of start_server.sh shows its current content
        # (newer codex traces omit file-write tool calls entirely)
        seen = _full_read_of_server_file(tool, cmd, path, ti, out)
        if seen is not None:
            add_version(st.i, seen)
        if tool in ("write", "edit") and path.endswith("evaluate.py"):
            log.eval_script_modified = True
        if is_shell and EVAL_TAMPER_RE.search(cmd):
            log.eval_script_modified = True

        if is_shell:
            launches_eval = _launches_eval(cmd)
            if EVAL_KILL_RE.search(cmd):
                log.n_eval_kills += 1
                if log.evals and not log.evals[-1].observed:
                    log.evals[-1].killed = True
            # -- server (re)starts: a launch command, or a startup banner in the output
            started_by_cmd = SERVER_START_RE.search(cmd) and not READ_ONLY_RE.match(cmd) and "--help" not in cmd \
                and not re.search(r"^\s*(cat|sed|head|grep|chmod|ls|vim|nano|bash\s+-n)\b[^|;&]*" + SERVER_FILE, cmd)
            started_by_output = STARTUP_BANNER_RE.search(out) and not re.match(r"\s*(tail|cat|grep|less|head)\b", cmd)
            if started_by_cmd or started_by_output:
                live_idx = cur_idx()
                evals_on_instance = 0
                log.server_starts.append((st.i, live_idx))
            # -- eval launches
            if launches_eval:
                log.evals.append(EvalLaunch(
                    step_i=st.i, quick=bool(QUICK_RE.search(cmd)), config_idx=live_idx,
                    config_stale=(cur_idx() is not None and live_idx is not None and cur_idx() != live_idx),
                    minute=elapsed, error=st.is_error, standard=_is_standard_eval(cmd), warm=evals_on_instance > 0))
                evals_on_instance += 1

        # -- metric observations (from any output the agent saw)
        if tool in ("bash", "shell", "taskoutput", "monitor", "read") and out and OBS_KW_RE.search(out):
            pending = log.evals[-1].config_idx if log.evals else None
            obs = _extract_observation(out, st.i, pending)
            if obs:
                obs.quick = log.evals[-1].quick if log.evals else None
                obs.standard = log.evals[-1].standard if log.evals else None
                obs.warm = log.evals[-1].warm if log.evals else None
                log.observations.append(obs)
                if log.evals:
                    log.evals[-1].observed = True

        # -- codex: the appended cumulative diff is the workspace state *after* this command
        cd = st.codex_diff
        if cd:
            rec = _content_from_codex_diff(cd)
            if rec is not None:
                add_version(st.i, rec)
            if re.search(r"diff --git a/+\S*evaluate\.py", cd):
                log.eval_script_modified = True
        if log.eval_script_modified and log.eval_script_step is None:
            log.eval_script_step = st.i
    return log
