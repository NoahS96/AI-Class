#!/usr/bin/python3
#Written by Andrew Schaffer, ivp811

from chatbot import chatbot

print('******************************')
print('  Welcome to Rowdy Messanger')
print('******************************')
print('<<<type "bye" to exit>>>')

# Initialize the chatbot class
bot = chatbot()

# Continue accepting user input and passing to bot until 'bye' is entered
while True:
    sUserMessage = input("user: ")
    sBotResponse = bot.respondTo(sUserMessage)
    
    print("bot: %s" % (sBotResponse))

    if sUserMessage == "bye":
        exit(0)
