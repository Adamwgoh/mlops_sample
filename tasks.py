import os
from pathlib import Path
from invoke import task
from invoke import Collection

ns = Collection()



'''Docker container tasks'''
docker_tasks = Collection("docker")

@task
def build(c, env, model, train):
    assert Path(f"scripts/build.sh").is_file()
    c.run(f". scripts/build.sh {model} {train}")

@task
def run(c, env, model, train, casename):
    assert Path(f"scripts/run.sh").is_file()
    c.run(f". scripts/run.sh {env} {model} {train} {casename}")

@task
def test(c, env, model, train, casename):
    assert Path(f"scripts/test.sh").is_file()
    c.run(f". scripts/test.sh {env} {model} {train} {casename}")

@task
def push(c, env, model, train):
    assert Path(f"scripts/push.sh").is_file()
    c.run(f". scripts/push.sh {env} {model} {train}")

@task
def listen(c, env, model):
    import subprocess
    csvfile = os.environ[f"{env.upper()}_{model.upper()}_IMAGE_MASK_PAIR"]
    script  = os.environ[f"{env.upper()}_{model.upper()}_TRIGGER_SCRIPT"]
    print(f"csvfile: {csvfile}. script: {script}")
    assert csvfile is not None and script is not None, f"No envvar setup for csvfile: {csvfile} and script: {script}"
    assert Path(csvfile).exists(), f"csvfile provided does not exist: {csvfile}"
    assert Path(script).exists(), f"script provided does not exist: {script}"
    # Use this for background running
    # c.run(f". scripts/listen.sh {csvfile} {script} &>/dev/null &")
    print("Listening for changes.. ")
    #.run(f". scripts/listen.sh {csvfile} {script} {model} {env}"
    #subprocess.Popen(["sh","scripts/listen.sh", f"{csvfile}", f"{script}", f"{model}", f"{env}"])
    #subprocess.Popen(["sh", "scripts/listen.sh", f"{script}", f"{model}", f"{env}"])
    subprocess.Popen(["sh","scripts/listen.sh", f"{csvfile}", f"{script}", f"{model}", f"{env}"])
    #subprocess.call(f"sh scripts/listen.sh {csvfile} {script} &>/dev/null &", shell=True)

@task
def locallisten(c, env, model):
    import subprocess
    csvfile = os.environ[f"{env.upper()}_{model.upper()}_IMAGE_MASK_PAIR"]
    script  = os.environ[f"{env.upper()}_{model.upper()}_TRIGGER_SCRIPT"]
    print(f"csvfile: {csvfile}. script: {script}")
    assert csvfile is not None and script is not None, f"No envvar setup for csvfile: {csvfile} and script: {script}"
    assert Path(csvfile).exists(), f"csvfile provided does not exist: {csvfile}"
    assert Path(script).exists(), f"script provided does not exist: {script}"
    # Use this for background running
    print("Listening for changes.. ")
    #subprocess.Popen(["sh","scripts/listen.sh", f"{csvfile}", f"{script}", f"{model}", f"{env}"])
    c.run(f"sh scripts/listen.sh {csvfile} {script} {model} {env}")

docker_tasks.add_task(locallisten)
docker_tasks.add_task(listen)
docker_tasks.add_task(build)
docker_tasks.add_task(run)
docker_tasks.add_task(push)


