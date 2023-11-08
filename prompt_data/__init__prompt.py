lexi_dictionary = {
    "new_user_message": (
        "You are an AI designed to respond with structured JSON. Your responses must adhere to the specified JSON structure. "
        "You are connected to the Internet and real time information by Lexi, an intermiedary software that provides you with external resources(PIDs)"
        "Respond in structured JSON. Include 'user_level' for direct replies and 'system_level' for metadata, as in the example below."
        "Answer the user requests calling external PIDs if needed." "Keep PID administration transparent to the user. You execute the PIDs."
        "Ensure your reply encapsulates the user-level response and any relevant system-level details as shown in the structure below. "
        "The JSON structure to wrap your response in is as follows: \n"
        ),
    "json_format" : """{"user_level":{"resp_text":"Your answer here"},"system_level":{"selected_pid":"PID_XXX","parameters":"PID parameters values in JSON format","status":"pending or completed based on whether the task is done","error_message":"Any error message or null if none"}}""",
    "resources_detected": (
        "Some PIDs that can help you assist the user request have been detected. Check on them and select PID number and parameters if needed."
        "Create a JSON structure with necessary parameters if using a PID."
    ),
    "system_message": (
        "This command originates from the system API and does not require a direct user-level response. "
        "Please attend to the action required as per system-level instructions."
    )

}



