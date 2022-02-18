import lineman_chat
import random

def main_lineman(sentence):
  while True:
#     sentence = input("user: ")
    if sentence == 'q':
      break
    else:
      keyword_tag = lineman_chat.lineman_final_process(sentence)
      # get keyword_tag in word for find responses in tag with random answer.
      for word in keyword_tag:
        for intent in lineman_chat.lineman_data['intents']:
          # print(word)
          if word == intent['tag']:
            result =  random.choice(intent['responses'])
            break
        print(keyword_tag)
        return result


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def main():
    return render_template("main.html")
@app.route('/lineman')
def lineman():
        return render_template("lineman.html")
@app.route("/get")
def get_response_lineman():
    userText = request.args.get('msg')
    return main_lineman(userText)

if __name__ == '__main__':
    app.run(debug=True)