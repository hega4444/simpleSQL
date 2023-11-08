#Protocols for external calls

pid_protocol = {}

pid_protocol[101] = {
        'name': 'bing_search',
        'show_return_to_user': True,
        'ERROR': {
            'sys_prompt': "There was an error executing the Bing search. Please check the parameters and try again.",
            'user_message': "I'm sorry, but I wasn't able to retrieve information from Bing search due to an error. Could you please specify your request further or try again later?",
        },
        'BEFORE':{
            'user_message': "Im retrieving information from the Internet right now. Please wait a few moments.",
        },
        'AFTER': {
            'sys_prompt': ("This is the information the user requested (has already been printed on screen). Offer the user additional help on analyzing the information. Do not show this info back again to the user."
                        "Use 'pid_0002' to read any NID_xxxx to further process the information."
                        "Only you know about the NIDs ids, user has no knowledge. Do not mention NIDs to the user."
                    ),
            'user_message': "I've found this information, give me a few moments to process it too.",
            'user_input' : "Please let me know if I can provide further assist you on your request."
        },
        'NEXT':{
            'call': "extract_article", #requests a second step.
            }    
    }

pid_protocol[102] = {
        'name': 'extract_article',
        'show_return_to_user' : True,
        'ERROR': {
            'sys_prompt': "There was an error executing the Bing search. Please check the parameters and try again.",
            'user_message': "I'm sorry, but I wasn't able to retrieve information from Bing search due to an error. Could you please specify your request further or try again later?",
        },
        'BEFORE':{
            'user_message': "Im extracting the contents of the article right now, just a moment.",
        },
        'AFTER': {
            'sys_prompt': "Information extracted from NID is being included in the attached data. Use it to assist user request. Protocol for function finishes here.",
            'user_message': "Here's an extract from the article:",
        },
        'NEXT': None    #no other resources needed
        
    }