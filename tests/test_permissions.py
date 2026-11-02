from pathlib import Path

import pytest

from tinyharness.hooks import Hook, HookRunner
from tinyharness.permissions import Decision, PermissionGate, PermissionMode, PermissionRules, Rule
from tinyharness.tools.agent import AgentTool
from tinyharness.tools.read import ReadTool
from tinyharness.tools.write import WriteTool
from tinyharness.workspace import Scratch, Workspace

READ, WRITE = ReadTool(), WriteTool()


class User:
    """A scripted permission callback that records what it was asked."""

    def __init__(self, *answers: Decision) -> None:
        self.answers = list(answers)
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        return self.answers.pop(0) if self.answers else Decision(True, "ok", "user")


YES = Decision(True, "ok", "user")
REMEMBER = Decision(True, "ok", "user", remember=True)
NO = Decision(False, "no", "user")


def gate(tmp_path, mode=PermissionMode.ASK, **kwargs) -> PermissionGate:
    return PermissionGate(workspace=Workspace(tmp_path), mode=mode, **kwargs)


# ---- what needs asking ----


async def test_reads_inside_the_project_never_prompt(tmp_path):
    user = User()
    d = await gate(tmp_path, callback=user).check(READ, {}, "src/app.py")
    assert d.allow and d.source == "mode"
    assert user.requests == []


async def test_side_effect_free_tools_without_a_path_never_prompt(tmp_path):
    agent_tool = AgentTool({}, spawner=None)
    assert (await gate(tmp_path).check(agent_tool, {}, None)).allow


async def test_reads_outside_the_project_prompt(tmp_path):
    user = User(YES)
    d = await gate(tmp_path, callback=user).check(READ, {}, "/etc/hosts")
    assert d.allow and len(user.requests) == 1


async def test_edits_prompt_in_default_mode(tmp_path):
    user = User(YES)
    await gate(tmp_path, callback=user).check(WRITE, {}, "a.py", summary="Write(a.py, 1 lines)")
    [request] = user.requests
    assert request.summary == "Write(a.py, 1 lines)"
    assert request.remember_label == "allow all edits in this project for this session"


async def test_no_callback_means_deny(tmp_path):
    d = await gate(tmp_path).check(WRITE, {}, "a.py")
    assert not d.allow and d.source == "prompt"


# ---- modes ----


async def test_accept_edits_only_inside_the_project(tmp_path):
    g = gate(tmp_path, PermissionMode.DEFAULT)
    assert (await g.check(WRITE, {}, "in.txt")).source == "mode"
    assert (await g.check(WRITE, {}, "/tmp/elsewhere")).source == "prompt"


async def test_bypass_allows_everything(tmp_path):
    assert (await gate(tmp_path, PermissionMode.BYPASS).check(WRITE, {}, "/etc/x")).allow


async def test_plan_blocks_writes(tmp_path):
    d = await gate(tmp_path, PermissionMode.PLAN, callback=User(YES)).check(WRITE, {}, "x.txt")
    assert not d.allow and d.source == "mode"


def test_shift_tab_cycles_default_auto_and_plan():
    """Ask mode is out of the cycle (`/mode ask`): Shift+Tab from it goes back to default."""
    mode = PermissionMode.ASK
    seen = [mode := mode.next() for _ in range(4)]
    assert seen == [
        PermissionMode.DEFAULT,
        PermissionMode.AUTO,
        PermissionMode.PLAN,
        PermissionMode.DEFAULT,
    ]


# ---- "don't ask again" ----


async def test_remembering_an_edit_switches_to_accept_edits(tmp_path):
    user = User(REMEMBER)
    g = gate(tmp_path, callback=user)
    await g.check(WRITE, {}, "src/a.py")
    assert g.mode == PermissionMode.DEFAULT
    assert (await g.check(WRITE, {}, "tests/b.py")).allow  # different dir, same tool family
    assert len(user.requests) == 1


async def test_remembering_an_outside_path_covers_that_directory(tmp_path_factory):
    project, outside = tmp_path_factory.mktemp("project"), tmp_path_factory.mktemp("downloads")
    user = User(REMEMBER)
    g = gate(project, callback=user)
    assert user.answers  # sanity
    await g.check(READ, {}, str(outside / "a.pdf"))
    assert user.requests[0].remember_label == f"allow reading in {outside}/ for this session"
    d = await g.check(READ, {}, str(outside / "b.pdf"))
    assert d.allow and d.source == "rule"
    assert g.mode == PermissionMode.ASK  # outside paths never flip the mode


async def test_plain_yes_is_not_remembered(tmp_path):
    user = User(YES, YES)
    g = gate(tmp_path, callback=user)
    await g.check(WRITE, {}, "a.py")
    await g.check(WRITE, {}, "a.py")
    assert len(user.requests) == 2


async def test_denials_are_not_remembered(tmp_path):
    user = User(NO, YES)
    g = gate(tmp_path, callback=user)
    assert not (await g.check(WRITE, {}, "a.py")).allow
    assert (await g.check(WRITE, {}, "a.py")).allow


# ---- configured rules ----


@pytest.mark.parametrize(
    ("rule", "tool", "path", "expected"),
    [
        ("Edit", "Edit", "/p/any.py", True),
        ("Edit", "Write", "/p/any.py", False),
        ("Edit(src/*)", "Edit", "/p/src/deep/x.py", True),
        ("Edit(src/*)", "Edit", "/p/tests/x.py", False),
        ("*(/etc/*)", "Read", "/etc/hosts", True),
        ("Read(*.env)", "Read", "/p/.env", True),
    ],
)
def test_rule_matching(rule, tool, path, expected):
    from pathlib import Path

    assert Rule.parse(rule, Path("/p")).matches(tool, path) is expected


def test_windows_rule_paths_match_with_either_separator():
    """Seen in CI: remembered and configured rules used different Windows separators."""
    rule = Rule("Read", "C:/Users/example/Downloads/*")
    assert rule.matches("Read", r"C:\Users\example\Downloads\report.pdf")


def test_invalid_rule_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Rule.parse("Edit(src/*", tmp_path)


async def test_config_rules_allow_and_deny_beats_allow(tmp_path):
    rules = PermissionRules.parse(allow=["Write"], deny=["Write(secrets/*)"], root=tmp_path)
    g = gate(tmp_path, rules=rules, callback=User())
    assert (await g.check(WRITE, {}, "notes.md")).source == "rule"
    d = await g.check(WRITE, {}, "secrets/key.txt")
    assert not d.allow and d.source == "rule"


async def test_deny_rules_apply_even_to_reads(tmp_path):
    rules = PermissionRules.parse(allow=[], deny=["Read(private/*)"], root=tmp_path)
    assert not (await gate(tmp_path, rules=rules).check(READ, {}, "private/x.md")).allow


# ---- other layers ----


async def test_scratch_dir_is_writable_without_asking(tmp_path):
    scratch = Scratch(tmp_path, "s1")
    d = await gate(tmp_path, scratch=scratch).check(WRITE, {}, str(scratch.dir / "notes.md"))
    assert d.allow


async def test_skill_assets_are_readable_without_asking(tmp_path):
    project, skill_dir = tmp_path / "proj", tmp_path / "skill"
    project.mkdir()
    skill_dir.mkdir()
    g = PermissionGate(workspace=Workspace(project, read_roots=[skill_dir]))
    assert (await g.check(READ, {}, str(skill_dir / "ref.md"))).allow


async def test_hook_deny_beats_bypass(tmp_path):
    hooks = HookRunner([Hook(event="PreToolUse", type="python", callable=lambda e, p: {"decision": "deny"})])
    d = await gate(tmp_path, PermissionMode.BYPASS, hooks=hooks).check(WRITE, {}, "x.txt")
    assert not d.allow and d.source == "hook"


# ---- a plain "yes" to a read covers the same read later ----


async def test_yes_to_a_read_is_remembered_for_that_file_only(tmp_path):
    user = User(YES, YES)
    g = gate(tmp_path, callback=user)
    assert (await g.check(READ, {}, "/etc/hosts")).allow
    assert (await g.check(READ, {}, "/etc/hosts")).allow
    assert len(user.requests) == 1  # not asked twice for the same file
    await g.check(READ, {}, "/etc/resolv.conf")
    assert len(user.requests) == 2  # another file still asks


async def test_yes_to_an_edit_is_not_remembered(tmp_path):
    user = User(YES, YES)
    g = gate(tmp_path, callback=user)
    await g.check(WRITE, {}, "a.py")
    await g.check(WRITE, {}, "a.py")
    assert len(user.requests) == 2  # each change is shown for review


# ---- auto mode ----


class Classifier:
    """Records what reaches the classifier; answers with a fixed verdict."""

    def __init__(self, verdict="allow"):
        from tinyharness.permissions.classifier import AutoModeClassifier

        self.verdict = verdict
        self.seen = []
        self.fast_filter = AutoModeClassifier.fast_filter.__get__(self)

    async def classify(self, tool, args, history_tail="", note=""):
        from tinyharness.permissions.classifier import ClassifierDecision

        self.seen.append(tool)
        return ClassifierDecision(self.verdict, "judged")


def auto(tmp_path, user=None, verdict="allow"):
    classifier = Classifier(verdict)
    scratch = Scratch(tmp_path, "s")
    g = gate(tmp_path, PermissionMode.AUTO, callback=user or User(), classifier=classifier, scratch=scratch)
    return g, classifier


async def test_auto_mode_does_routine_project_work_without_asking_anyone(tmp_path):
    from tinyharness.tools.edit import EditTool

    user = User()
    g, classifier = auto(tmp_path, user)
    (tmp_path / "small.py").write_text("x = 1\n")
    assert (await g.check(WRITE, {"path": "new.py", "content": "print(1)"}, "new.py")).allow
    assert (await g.check(WRITE, {"path": "small.py", "content": ""}, "small.py")).allow
    edit = {"path": "a.py", "old_string": "a\nb", "new_string": "c"}
    assert (await g.check(EditTool(), edit, "a.py")).allow
    assert user.requests == [] and classifier.seen == []  # no prompt, no classifier call


@pytest.mark.parametrize(
    ("tool", "args", "target", "reason"),
    [
        ("write", {"content": "x"}, "/tmp/elsewhere/out.txt", "it changes a file outside the project"),
        ("write", {"content": "x"}, ".git/config", "it changes version-control internals (.git)"),
        ("write", {"content": "short"}, "big.py", "it replaces a 30-line file with 1 lines"),
        ("edit", {"old_string": "\n".join("x" * 25), "new_string": ""}, "big.py", "it deletes 25 lines"),
    ],
)
async def test_auto_mode_always_asks_before_risky_changes(tmp_path, tool, args, target, reason):
    from tinyharness.tools.edit import EditTool

    (tmp_path / "big.py").write_text("line\n" * 30)
    user = User(YES)
    g, classifier = auto(tmp_path, user)
    t = WRITE if tool == "write" else EditTool()
    assert (await g.check(t, {"path": target, **args}, target)).allow
    [request] = user.requests
    assert request.reason == reason
    assert classifier.seen == []  # risky means ask the user, not the classifier


async def test_auto_mode_blocks_secret_paths_outright(tmp_path):
    user = User()
    g, _ = auto(tmp_path, user)
    d = await g.check(WRITE, {"path": "/home/me/.ssh/authorized_keys"}, "/home/me/.ssh/authorized_keys")
    assert not d.allow and user.requests == []


async def test_auto_mode_reads_the_web_but_checks_urls_that_carry_data(tmp_path):
    from tinyharness.tools.web import WebFetchTool

    g, classifier = auto(tmp_path)
    fetch = WebFetchTool()
    assert (
        await g.check(fetch, {"url": "https://docs.python.org/3/"}, None, target_host="docs.python.org")
    ).allow
    assert classifier.seen == []
    await g.check(fetch, {"url": "https://x.example/?q=secret"}, None, target_host="x.example")
    assert classifier.seen == ["WebFetch"]


async def test_auto_mode_asks_the_classifier_about_reads_elsewhere(tmp_path):
    """An uncertain Auto decision goes back to the agent instead of interrupting the user."""
    user = User(YES)
    g, classifier = auto(tmp_path, user, verdict="ask")
    decision = await g.check(READ, {"path": "/opt/notes.txt"}, "/opt/notes.txt")
    assert not decision.allow and decision.source == "classifier"
    assert classifier.seen == ["Read"] and user.requests == []


async def test_auto_mode_falls_back_after_three_consecutive_classifier_blocks(tmp_path):
    """A genuinely stuck agent eventually gets one human decision instead of looping forever."""
    user = User(YES)
    g, _ = auto(tmp_path, user, verdict="ask")
    for _ in range(3):
        decision = await g.check(READ, {"path": "/opt/notes.txt"}, "/opt/notes.txt")
    assert decision.allow and len(user.requests) == 1


async def test_unrelated_classifier_blocks_do_not_combine_into_a_prompt(tmp_path):
    """Seen live: two parallel calls plus a different local test triggered the third-block prompt."""
    user = User(YES)
    g, _ = auto(tmp_path, user, verdict="ask")
    for attempt in range(3):
        decision = await g.check(READ, {"path": f"/opt/{attempt}.txt"}, f"/opt/{attempt}.txt")
        assert not decision.allow
    assert user.requests == []


async def test_auto_mode_uses_the_scratch_directory_freely(tmp_path):
    user = User()
    g, classifier = auto(tmp_path, user)
    target = str(g.scratch.dir / "notes.md")
    assert (await g.check(WRITE, {"path": target, "content": "x"}, target)).allow
    assert user.requests == [] and classifier.seen == []


async def test_risky_changes_in_auto_mode_are_judged_one_at_a_time(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("x\n" * 100)
    user = User(REMEMBER, REMEMBER)
    g = gate(tmp_path, PermissionMode.AUTO, callback=user)
    for _ in range(2):
        assert (await g.check(WRITE, {"content": "y"}, "big.py")).allow
    first, _ = user.requests  # asked both times
    assert first.reason.startswith("it replaces a 100-line file") and first.remember_label == ""
    assert g.mode == PermissionMode.AUTO  # still auto: the answer didn't switch to acceptEdits


# ---- subagents follow the session's mode ----


@pytest.mark.parametrize(
    ("session", "limit", "runs_in"),
    [
        ("auto", None, "auto"),  # no limit: the session's mode
        ("auto", "default", "default"),  # a limit only ever makes it stricter...
        ("ask", "default", "ask"),  # ...never more permissive than the session
        ("ask", "bypassPermissions", "ask"),
        ("plan", None, "plan"),
        ("bypassPermissions", "default", "default"),
    ],
)
def test_a_subagent_runs_in_the_stricter_of_the_session_and_its_limit(tmp_path, session, limit, runs_in):
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.parse(session))
    child = gate.child("helper", PermissionMode.parse(limit) if limit else None)
    assert child.mode == PermissionMode.parse(runs_in)


def test_switching_mode_reaches_subagents_already_running(tmp_path):
    """Switching to auto mid-run used to leave running subagents asking as before."""
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.ASK)
    free, capped = gate.child("general-purpose", None), gate.child("reviewer", PermissionMode.DEFAULT)
    gate.set_mode(PermissionMode.AUTO)
    assert (gate.mode, free.mode, capped.mode) == (
        PermissionMode.AUTO,
        PermissionMode.AUTO,
        PermissionMode.DEFAULT,
    )


async def test_allowing_all_edits_at_a_prompt_reaches_subagents_too(tmp_path):
    from tinyharness.tools.edit import EditTool

    async def yes_dont_ask_again(request):
        return Decision(True, "ok", "user", remember=True)

    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.ASK, callback=yes_dont_ask_again)
    child = gate.child("general-purpose", None)
    await gate.check(EditTool(), {}, "src/app.py")  # "allow all edits in this project"
    assert gate.mode == child.mode == PermissionMode.DEFAULT


async def test_a_subagents_prompt_says_whose_it_is(tmp_path):
    from tinyharness.tools.edit import EditTool

    asked = []

    async def user(request):
        asked.append(request)
        return Decision(True, "ok", "user")

    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.ASK, callback=user)
    await gate.child("general-purpose", None).check(EditTool(), {}, "src/app.py")
    assert asked[0].agent == "general-purpose"


def test_mode_changes_are_logged(tmp_path, monkeypatch):
    import json

    from tinyharness import Session, SessionOptions

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path))
    try:
        session.set_mode("auto")
        records = [json.loads(line) for line in session.transcript.path.read_text().splitlines()]
        assert {"event": "mode", "mode": "auto"}.items() <= records[-1].items()
    finally:
        session.close()


# ---- reading tools share one answer ----


async def test_allowing_reading_in_a_folder_covers_every_reading_tool(tmp_path_factory):
    """Read, then Grep, then Glob on ~/Downloads used to be three questions."""
    from tinyharness.tools.glob import GlobTool
    from tinyharness.tools.grep import GrepTool

    project, outside = tmp_path_factory.mktemp("project"), tmp_path_factory.mktemp("downloads")
    (outside / "deck.html").write_text("<html>")
    user = User(REMEMBER)
    g = gate(project, callback=user)
    await g.check(READ, {}, str(outside / "deck.html"))
    assert (await g.check(GrepTool(), {}, str(outside / "deck.html"))).allow
    assert (await g.check(GlobTool(), {}, str(outside))).allow  # the folder itself
    assert len(user.requests) == 1


async def test_a_yes_to_reading_a_file_covers_grepping_it(tmp_path):
    from tinyharness.tools.grep import GrepTool

    user = User(YES)
    g = gate(tmp_path, callback=user)
    await g.check(READ, {}, "/etc/hosts")
    assert (await g.check(GrepTool(), {}, "/etc/hosts")).allow
    assert len(user.requests) == 1


async def test_other_tools_still_get_their_own_answer(tmp_path_factory):
    project, outside = tmp_path_factory.mktemp("project"), tmp_path_factory.mktemp("downloads")
    user = User(REMEMBER, YES)
    g = gate(project, callback=user)
    await g.check(READ, {}, str(outside / "a.txt"))
    await g.check(WRITE, {}, str(outside / "b.txt"))  # reading approval doesn't cover writing
    assert len(user.requests) == 2


async def test_default_mode_works_inside_the_project_and_asks_beyond_it(tmp_path):
    """The default mode doesn't ask for routine work: edits in the project and sandboxed
    commands run. Leaving the boundary (another folder) still asks. `ask` is the strict mode,
    and the earlier name `acceptEdits` means default."""
    from tinyharness.tools.write import WriteTool

    asked = []

    async def user(request):
        asked.append(request.tool)
        return Decision(False, "no", "user")

    inside = await gate(tmp_path, PermissionMode.DEFAULT, callback=user).check(
        WriteTool(), {"path": "a.py", "content": "x"}, target_path=tmp_path / "a.py"
    )
    outside = await gate(tmp_path, PermissionMode.DEFAULT, callback=user).check(
        WriteTool(), {"path": "/elsewhere/a.py", "content": "x"}, target_path=Path("/elsewhere/a.py")
    )
    strict = await gate(tmp_path, PermissionMode.ASK, callback=user).check(
        WriteTool(), {"path": "a.py", "content": "x"}, target_path=tmp_path / "a.py"
    )
    assert inside.allow and not outside.allow and not strict.allow and asked == ["Write", "Write"]
    assert PermissionMode.parse("acceptEdits") == PermissionMode.DEFAULT


async def test_the_auto_mode_classifier_is_tagged_with_its_own_role():
    """Seen live: the classifier's calls went out as the lead's, so a gateway counted a permission
    check on the small model as a lead turn (and gave it the lead's thinking)."""
    from tinyharness.llm import identity
    from tinyharness.permissions.classifier import AutoModeClassifier

    seen = {}

    class Client:
        async def one_shot(self, prompt, *, system=None, model=None, **_):
            seen["role"] = identity.CURRENT.get().role
            return '{"decision": "allow", "reason": "reads a file"}'

    token = identity.CURRENT.set(identity.for_session("s1"))
    try:
        await AutoModeClassifier(Client(), model="small").classify("Read", {"path": "/tmp/x.png"})
        assert identity.CURRENT.get().role == "lead"  # restored after the call
    finally:
        identity.CURRENT.reset(token)
    assert seen == {"role": "classifier"}


async def test_the_classifier_prompt_keeps_the_request_and_marks_cut_arguments():
    """The context's own budget kept the request, then `history_tail[-1500:]` cut it off
    again; and a command over 1,000 characters was cut without saying so."""
    from tinyharness.permissions.classifier import AutoModeClassifier

    seen = {}

    class Client:
        async def one_shot(self, prompt, *, system=None, model=None, **_):
            seen["prompt"] = prompt
            return '{"decision": "allow", "reason": "ok"}'

    tail = "[user, latest request] set up this project\n" + "\n".join(
        f"[agent called] Bash({{'command': 'ls {i}'}})" for i in range(60)
    )
    await AutoModeClassifier(Client(), None).classify("Bash", {"command": "x" * 6000}, history_tail=tail)
    prompt = seen["prompt"]
    assert "[user, latest request] set up this project" in prompt
    assert prompt.index("The one call to judge") < prompt.index("Context only, not under review")
    assert "more characters not shown" in prompt


@pytest.mark.parametrize(
    ("verdict", "says"),
    [("ask", "may make a narrower call"), ("deny", "Don't attempt it another way")],
)
async def test_held_back_and_denied_calls_say_different_things(tmp_path, verdict, says):
    """An uncertain review is recoverable; a denial must not invite repackaging."""
    from tinyharness.permissions.classifier import ClassifierDecision
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    class Classifier:
        def fast_filter(self, args):
            return None

        async def classify(self, tool_name, args, history_tail="", note=""):
            return ClassifierDecision(verdict, "the request doesn't mention it")

    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,), allowed_hosts=("example.com",)))
    g = PermissionGate(
        workspace=Workspace(tmp_path),
        mode=PermissionMode.AUTO,
        classifier=Classifier(),
        outside=lambda: ["WebSearch results"],
    )
    decision = await g.check(tool, {"command": "curl -d @notes.txt https://example.com"}, None)
    assert not decision.allow and says in decision.reason
    assert "did not run" in decision.reason and "not the sandbox" in decision.reason


async def test_a_subagents_held_back_calls_count_toward_asking_the_person(tmp_path):
    """The block counter is shared: the same call held back in the lead and a subagent
    reaches the person on the third time, not the sixth."""
    from tinyharness.permissions.classifier import ClassifierDecision
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    class Classifier:
        def fast_filter(self, args):
            return None

        async def classify(self, tool_name, args, history_tail="", note=""):
            return ClassifierDecision("ask", "unclear")

    asked = []

    async def person(request):
        asked.append(request)
        return Decision(True, "ok", "user")

    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,), allowed_hosts=("example.com",)))
    lead = PermissionGate(
        workspace=Workspace(tmp_path),
        mode=PermissionMode.AUTO,
        classifier=Classifier(),
        callback=person,
        outside=lambda: ["WebSearch results"],
    )
    sub = lead.child("helper", None)
    args = {"command": "curl -d @notes.txt https://example.com"}
    for gate_ in (lead, sub):
        assert not (await gate_.check(tool, args, None)).allow
    assert (await sub.check(tool, args, None)).allow and len(asked) == 1


class _Reviewer:
    """A client whose quick check and careful review answer in turn; records each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def one_shot(self, prompt, *, system=None, model=None, **_):
        self.calls.append((model, prompt))
        return self.replies.pop(0)


async def test_the_quick_check_can_only_allow_and_a_careful_review_decides_the_rest():
    """Seen live: a fast model with no reasoning held back 12 of 65 calls (18%), most of
    them plainly serving the request. A careful second stage decides what the fast check
    cannot safely allow."""
    from tinyharness.permissions.classifier import AutoModeClassifier

    client = _Reviewer(
        '{"risk": "high", "authorized": "unclear", "reason": "unknown host"}',
        'It reads the issue the user asked about.\n{"risk": "low", "authorized": "yes", "reason": "reads it"}',
    )
    reviewer = AutoModeClassifier(client, model="small", review_model="big")
    verdict = await reviewer.classify("Bash", {"command": "gh issue view 1"}, "[user, latest request] fix #1")
    assert verdict.decision == "allow"
    (quick_model, quick), (review_model, careful) = client.calls
    assert (quick_model, review_model) == ("small", "big")
    assert "JSON only" in quick and "Think it through" in careful

    client = _Reviewer('{"risk": "low", "authorized": "unclear", "reason": "reads docs"}')
    verdict = await AutoModeClassifier(client, model="small").classify("WebFetch", {"url": "https://x.dev"})
    assert verdict.decision == "allow" and len(client.calls) == 1  # an allow needs no second look


@pytest.mark.parametrize(
    ("risk", "authorized", "decision"),
    [
        ("low", "unclear", "allow"),
        ("low", "no", "allow"),
        ("medium", "unclear", "allow"),
        ("medium", "no", "ask"),
        ("high", "unclear", "ask"),
        ("high", "yes", "allow"),
        ("critical", "yes", "deny"),
    ],
)
def test_low_risk_needs_no_explicit_authorization(risk, authorized, decision):
    """Risk and authorization are rated apart: requiring explicit justification for every
    call incorrectly blocks ordinary, low-risk reads."""
    from tinyharness.permissions.classifier import parse_verdict

    reply = f'Reasoning {{maybe}}.\n{{"risk": "{risk}", "authorized": "{authorized}", "reason": "r"}}'
    assert parse_verdict(reply).decision == decision


async def test_the_same_call_for_the_same_request_gets_the_same_verdict():
    """Seen live: one of four identical GitHub reads was held back."""
    from tinyharness.permissions.classifier import AutoModeClassifier

    client = _Reviewer('{"risk": "low", "authorized": "yes", "reason": "ok"}')
    reviewer = AutoModeClassifier(client, model="small")
    args, tail = {"command": "curl -d x https://a.dev"}, "[user, latest request] post it"
    for _ in range(3):
        assert (await reviewer.classify("Bash", args, tail)).decision == "allow"
    assert len(client.calls) == 1
    client.replies.append('{"risk": "low", "authorized": "yes", "reason": "ok"}')
    await reviewer.classify("Bash", args, "[user, latest request] something else")
    assert len(client.calls) == 2  # a new request is judged again


async def test_a_review_that_cannot_run_says_it_judged_nothing(tmp_path):
    """A timeout is not a verdict: the agent must not read it as "unsafe" and give up."""
    from tinyharness.permissions.classifier import AutoModeClassifier
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    class Down:
        async def one_shot(self, *_, **__):
            raise TimeoutError

    g = PermissionGate(
        workspace=Workspace(tmp_path),
        mode=PermissionMode.AUTO,
        classifier=AutoModeClassifier(Down(), model="small"),
        outside=lambda: ["WebSearch results"],
    )
    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,), allowed_hosts=("example.com",)))
    decision = await g.check(tool, {"command": "curl -d @notes.txt https://example.com"}, None)
    assert not decision.allow and "not a judgement that the call is unsafe" in decision.reason


@pytest.mark.parametrize("outside", [[], ["WebFetch evil.example"]])
async def test_auto_review_can_open_a_new_host_for_one_command(tmp_path, outside):
    """A host the command needs is reviewed with the command and opened for that command
    alone. With outside content in the conversation the person decides:
    a new destination is how that content could get data sent out."""
    from tinyharness.permissions.classifier import ClassifierDecision
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    class Classifier:
        def fast_filter(self, args):
            return None

        async def classify(self, tool_name, args, history_tail="", note=""):
            assert "this one command only" in note
            return ClassifierDecision("allow", "the docs the user asked about")

    asked = []

    async def person(request):
        asked.append(request)
        return Decision(False, "no", "user")

    g = PermissionGate(
        workspace=Workspace(tmp_path),
        mode=PermissionMode.AUTO,
        classifier=Classifier(),
        callback=person,
        outside=lambda: outside,
    )
    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,), allowed_hosts=("pypi.org",)))
    args = {"command": "curl -s https://docs.example.dev/x", "network_host": "docs.example.dev"}
    decision = await g.check(tool, args, None, target_host="docs.example.dev")
    assert decision.allow is not outside and len(asked) == int(bool(outside))
    assert not g.session_network_hosts and not g.rules.allow  # nothing remembered


async def test_loopback_is_reviewed_even_with_outside_content_and_remembered(tmp_path):
    """Seen live: SGLang's engine needed loopback after a Hugging Face download, and the
    person was asked. Loopback never leaves the machine, so outside content can't use it to
    send data away; the review judges what local services it could reach."""
    from tinyharness.permissions.classifier import ClassifierDecision
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    seen = []

    class Classifier:
        def fast_filter(self, args):
            return None

        async def classify(self, tool_name, args, history_tail="", note=""):
            seen.append(note)
            return ClassifierDecision("allow", "runs the engine the user asked for")

    g = PermissionGate(
        workspace=Workspace(tmp_path),
        mode=PermissionMode.AUTO,
        classifier=Classifier(),
        callback=User(),
        outside=lambda: ["Bash downloads from huggingface.co"],
    )
    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,)), session_hosts=g.session_network_hosts)
    args = {"command": "python hello.py", "network_host": "localhost"}
    assert (await g.check(tool, args, None, target_host="localhost")).allow
    assert "loopback" in seen[0] and "localhost" in g.session_network_hosts
    assert (await g.check(tool, args, None, target_host="localhost")).allow and len(seen) == 1


async def test_one_approval_covers_a_services_download_hosts(tmp_path):
    """Seen live: a model download went from huggingface.co to cas-server.xethub.hf.co,
    one more blocked command and one more prompt. Approving the service opens its set."""
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.sandbox.proxy import service_hosts
    from tinyharness.tools.bash import BashTool

    assert "cas-server.xethub.hf.co" in service_hosts("huggingface.co")
    assert service_hosts("example.com") == ("example.com",)
    user = User(Decision(True, "ok", "user", remember=True))
    g = gate(tmp_path, PermissionMode.DEFAULT, callback=user)
    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,)), session_hosts=g.session_network_hosts)
    args = {"command": "python download.py", "network_host": "huggingface.co"}
    assert (await g.check(tool, args, None, target_host="huggingface.co")).allow
    request = user.requests[0]
    assert request.remember_label == "allow Hugging Face's 8 download hosts this session"
    assert "Hugging Face downloads use this exact destination set" in request.warning
    assert {"huggingface.co", "cas-server.xethub.hf.co"} <= g.session_network_hosts
