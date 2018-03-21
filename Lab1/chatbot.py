#!/usr/bin/python3
#Written by Andrew Schaffer, ivp811

class chatbot:

    # Dictionary of user input and their corresponding bot responses
    sdResponseDict = {  'hello':'Hi there', 'hey':'Hi', 'hi':'Ayy', 'how are you?':'Pretty good. What about you?',
                        'what\'s up?':'just talking with you!', 'cool!':'Not really', 'good':'Nice to hear',
                        'bye':'See ya next time'}
    # Default bot response if user input is not found in dictionary
    sDefaultResponse = 'lol what?'


    # Initialize the chatbot class
    def __init__(self):
        pass
    
    # responsdTo
    #   Parameters:
    #       String sMessage -   The user input to evaluate
    #   Purpose:
    #       Return an appropriate response to user input
    def respondTo(self, sMessage):
        
        # Save the response dictionary keys into a set as lowercase to make user input case-insensitive
        sdLowerCaseDict = set(k.lower() for k in self.sdResponseDict)

        # Check if the user input is in the dictionary. If so return the matching value, else return 
        # default message
        if sMessage.lower() in sdLowerCaseDict:
            return self.sdResponseDict[sMessage.lower()]
        else:
            return self.sDefaultResponse
