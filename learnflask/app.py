#!flask/bin/python
from flask import Flask
from flask import request
import json 

app = Flask(__name__)

file_r = open("./tasks.json", "r")
tasks = json.loads(file_r.read())
tasks_backup = tasks
file_r.close()
print(tasks)

#json db manipulation 
def read_tasks():
    file_r = open("./tasks.json", "r")
    tasks = json.loads(file_r.read())
    file_r.close()
    return tasks;

def write_task(id, title, description, done):
    try:
        file_r = open("./tasks.json", "r")
        tasks = json.loads(file_r.read())
        file_r.close()
        tasks_backup = tasks
        new_task = {"id": id, "title": title, "description": description, "done": done}
        tasks["tasks"].append(new_task)
        file_w = open("./tasks.json", "w")
        file_w.write(json.loads(tasks))
        file_w.close()
        print('finished posting')
        return 0
    except:
        file_w = open("./tasks.json", "w")
        file_w.write(json.dumps(tasks_backup))
        file_w.close()
        return -1
    return 0

def overwrite_tasks(tasks):
    file_w = open("./tasks.json", "w")
    file_w.write(json.dumps(tasks))
    file_w.close()


#api/ 
@app.route('/api')
def index():
    return "api home make requests!!"

#api/tasks GET
@app.route('/api/tasks', methods=["GET"])
def get_tasks():
    tasks = read_tasks()
    return json.dumps(tasks)
#api/tasks GET by ID
@app.route('/api/tasks/id/<int:id>', methods=["GET"])
def get_task_by_id(id):
    tasks = read_tasks()

    for task in tasks["tasks"]:
        if int(task["id"]) == id:
            return json.dumps(task)
    return "{\"msg\": \"not found\"}"

#api/tasks/post POST
@app.route("/api/tasks/id/<int:id>/title/<string:title>/description/<string:description>/done/<string:done>", methods = ["POST"])
def post_task(id, title, description, done):
    result = write_task(id, title, description, done)
    return "{\"msg\": \"success\"}"

#api/tasks PUT by ID
@app.route('/api/tasks/id/<int:id>', methods=["PUT"])
def put_task_by_id(id):
    tasks = read_tasks()

    for i in range(0, len(tasks["tasks"])):
        if int(tasks["tasks"][i]["id"]) == id:
            if(tasks["tasks"][i]["done"]=="True"):
                tasks["tasks"][i]["done"]="False"
            else:
                tasks["tasks"][i]["done"]="True"
            overwrite_tasks(tasks)
        return "{\"msg\": \"resource updated\"}"    
    
    return "{\"msg\": \"not found\"}"
#api/tasks PUT all tasks
@app.route('/api/tasks', methods=["PUT"])
def put_tasks():
    tasks = read_tasks()

    for i in range(0, len(tasks["tasks"])):
        if(tasks["tasks"][i]["done"]=="True"):
            tasks["tasks"][i]["done"]="False"
        else:
            tasks["tasks"][i]["done"]="True"
    overwrite_tasks(tasks)
    return "{\"msg\": \"resource updated\"}"    
    
    return "{\"msg\": \"not found\"}"

#api/tasks DELETE by ID
@app.route('/api/tasks/id/<int:id>', methods=["DELETE"])
def del_task_by_id(id):
    tasks = read_tasks()

    for i in range(0, len(tasks["tasks"])):
        if int(tasks["tasks"][i]["id"]) == id:
            del tasks["tasks"][i]
            overwrite_tasks(tasks)
        return "{\"msg\": \"resource updated\"}"    
    
    return "{\"msg\": \"not found\"}"
 
if __name__ == '__main__':
    app.run(debug=True)