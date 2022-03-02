from flask import Flask, request, jsonify, json, render_template, redirect

from transformers import pipeline


app = Flask(__name__)


# Load Model

def load_model():
    global classifier

    classifier = pipeline("zero-shot-classification",
        model="facebook/bart-large-mnli")


def get_model():
    global classifier

    return classifier



def evaluate(sequence,labels):

    # return {"result":{"labels":["food","religion"],"scores":[0.998341977596283,0.0016579453367739916],"sequence":"I love to eat"}}

    model = get_model()

    result = model(sequence, labels)



    return result


@app.route('/api/docs')
def api_doc_swagger():
    print('sending docs')
    return render_template('swaggerui.html')



@app.route('/api',methods = ['GET',"POST"])
def predict():
    error_list = []

    sequence = request.values.get('sequence')
    labels = request.values.getlist('labels')

    # Check if sequence exists
    if not sequence:
        error_list.append("Sequence variable not found")
    
    # Check if labels exist
    if not labels:
        error_list.append("Labels variable not found")
        

    if len(error_list) == 0:
        result = evaluate(sequence,labels)

        return jsonify({'result':result}),200

    else:
        return jsonify({'error':error_list}),400
            

    
    # return jsonify({'error':'Invalid Request'})


# @app.errorhandler(werkzeug.exceptions.BadRequest)
# def handle_bad_request(e):
#     return 'Internal Server Error!', 500
    
@app.errorhandler(404)
  
# inbuilt function which takes error as parameter
def not_found(e):
  
# defining function
  return redirect("/api/docs")

@app.route("/")
def home_page():

    return render_template('swaggerui.html')


    # return """Zero Shot Learning - API<br><br>
    #     <u>API Format</u><br>


    #     <h1>GET</h1><br>
    #     &lt;url&gt;/predict?sequence=<b>&lt;sequence&gt;</b>&labels=<b>&lt;label_1&gt;</b>&labels=<b>&lt;label_2&gt;</b><br><br>
    #     <h1>POST</h1><br>
    #     {"sequence":&lt;sequence&gt;, "labels":[&lt;label_1&gt;, &lt;label_2&gt;]"""




if __name__ == '__main__':

    load_model()


    app.run(host= '0.0.0.0',debug=False,port=9999)