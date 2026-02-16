from ai_adviser.api.main import app, health, ready


@app.get("/checks/liveness")
def liveness() -> dict[str, bool]:

    return health()  # type: ignore[no-any-return]


@app.get("/checks/readiness")
def readiness() -> dict[str, object]:

    return ready()  # type: ignore[no-any-return]
