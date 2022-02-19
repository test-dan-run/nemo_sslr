# read environment files (.env) and output a dictionary
def read_env(path: str) -> dict:

    variables = {}
    
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:

        # skip empty line
        if not line: continue
        # skip comments
        if line[0] == '#': continue

        # split by the 1st = character
        k, v = line.split('=', maxsplit= 1)
        k = k.strip('\r\n')
        v = v.strip('\r\n').lstrip()
        variables[k] = v

    if not variables:
        print('Environment dictionary is empty. Ignore this if running remotely. Otherwise check your environment filepath.')

    return variables

def generate_docker_task_string(docker_path: str, env_path: str) -> str:

    env_dict = read_env(env_path)
    docker_task_string = docker_path

    for k, v in env_dict.items():
        docker_task_string += f' --env {k}={v}'

    return docker_task_string
