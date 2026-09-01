# taper

A running simulator, built on published exercise-physiology research.

Oregon Trail for runners: you plan the training, the body responds, and the goal
race at the end of the block tells you whether you got it right. Push too hard
and you break down; push too little and you show up underprepared.

## Why it might actually work

Most sports games invent their resource system. This one does not have to.
Endurance physiology already *is* a resource-management game, and the tension is
exactly the interesting one:

> **Fitness only comes from stress. Stress accumulates fatigue. Fatigue is what
> breaks you.**

That is the Banister impulse-response model: two exponential decays with
different time constants. Fitness fades slowly, fatigue quickly. Which means the
taper falls out of the arithmetic for free -- cut volume for two weeks before the
race and your fatigue drains while your fitness barely moves. A player who works
that out on their own has discovered something real.

## What is science and what is game design

The training-response literature is solid. The **injury** literature is much
weaker than popular writing suggests: ACWR (acute:chronic workload ratio) has
been substantively criticised on methodological grounds, and the "10% per week"
rule has close to no support.

So this project draws the line explicitly. `physiology.py` holds published models
and cites each one at the point of use. The injury model, when it lands, will be
a hazard function *inspired* by the literature and tuned for game feel -- and it
will say so, in the code and in the game. A simulator that is straight about
which parts are evidence and which are design is more interesting than one that
pretends the whole thing is settled.

## Status

Early. The intake layer is done; the simulation engine is next.

```sh
uv run taper intake          # fill in the form at localhost:8000
uv run taper show profiles/you.json
```

The form asks for the runner, their recent training, race history, injury
history, life load and goal race, and derives what it can as you type: VDOT for
every race, peak VDOT per year as a career arc, a present-day fitness estimate
labelled by confidence, equivalent times at every standard distance, and
readiness flags against the goal. It runs on localhost and writes a JSON file.
No account, no network, nothing leaves the machine.

- `taper/athlete.py` -- the athlete profile: the schema the intake form fills
- `taper/physiology.py` -- Daniels-Gilbert VDOT, Riegel, Minetti gradient cost
- `taper/insights.py` -- what can be derived from a profile without simulating
- `taper/intake/` -- the local web form and the race-history importer
- `taper/profile_io.py` -- versioned JSON persistence

## Importing race history

There is no crawler, on purpose. A runner has a few dozen lifetime races, and
the timing industry is a dozen incompatible React apps whose athlete-search
endpoints are disallowed by `robots.txt` anyway. So the importer takes a paste
instead: select-all-copy the results off any timing site, or drop in a CSV, and
it pulls out distance, date, finish time and placing. One parser, every site,
nothing to break on a redesign. Whatever it gets wrong you fix in the table.

## Development

```sh
uv sync
uv run pytest
```

## License

MIT
