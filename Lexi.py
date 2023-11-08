import sys
sys.path.insert(0, './prompt_data/')

import openai
import inspect
import json
import feedparser
from typing import List
import requests
from urllib.parse import urlparse, parse_qs, unquote
from newspaper import Article
import os
import re
from __init__prompt import lexi_dictionary
import itertools


class LexiExternalCommand():
    # This class encapsulates the details for connecting an external process to the chat model, making it available to the user through NLP.

    FIX = 1
    ON_DEMMAND = 2

    _pid_counter = 0  # Class-level attribute acting as a counter

    def __init__(self, func, show_return_to_user = False, 
                 error= None, before= None, after= None, next= None, 
                 append_mode = ON_DEMMAND, printer = None):
        
        #PID settings
        self.call = func
        self.name = func.__name__

        type(self)._pid_counter += 1
        self.pid_id = "pid_" + str(type(self)._pid_counter).zfill(4)

        self.specs = self.function_specs(func)
        self.printer = printer
        self.append_mode = append_mode
        
        # Protocol settings
        self.ptc = {}
        self.ptc['error'] = error
        self.ptc['before'] = before
        self.ptc['after'] = after
        self.ptc['next'] = next
        self.ptc['show_return_to_user'] = show_return_to_user
        self.keys = ['error', 'before', 'after', 'next']
    
    def function_specs(self, func):
        # Checks the function to be appended to Lexi and retrieves all its specifications 

        sig = inspect.signature(func)
        
        return_annotation = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else "null"
        return_annotation = self.stringify_types(return_annotation)
        
        det_annotation = func.__annotations__
        
        params = {}
        for name, param in sig.parameters.items():
            new_param = {'req': 'y' if param.default == inspect.Parameter.empty else 'n'}
            if name in det_annotation:
                new_param['type'] = self.str_type(str(det_annotation[name]))
            params[name] = new_param

        source_lines = inspect.getsource(func).split('\n')

        #Parse header comments
        # Compile regular expressions for the comment sections
        keys_pattern = re.compile(r'#\s*KEYS:\s*(.*)')
        sum_pattern = re.compile(r'#\s*SUMM:\s*(.*)')

        # Initialize dictionary to store comment sections
        comment_sections = {
            'KEYS': None,
            'SUMM': None,
        }

        for line in source_lines:
                keys_match = keys_pattern.search(line)
                sum_match = sum_pattern.search(line)

                if keys_match:
                    comment_sections['KEYS'] = keys_match.group(1)
                elif sum_match:
                    comment_sections['SUMM'] = sum_match.group(1)
        
        func_specs = {
            'name': self.name,
            'summ': comment_sections['SUMM'],
            'keys' : comment_sections['KEYS'], 
            'params': params,
            'return': return_annotation,
        }

        return func_specs

    def stringify_types(self, data):
        #Convert type objects in a nested dictionary to their string representations.

        if isinstance(data, dict):
            return {k: self.stringify_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.stringify_types(item) for item in data]
        elif isinstance(data, type):
            return self.str_type(str(data))
        else:
            return data
    
    def str_type(self, text):
        if 'int' in text:
            return 'int'
        elif 'str' in text:
            return 'str'
        elif 'float' in text:
            return 'float'
        elif 'bool' in text:
            return 'bool'

    def update_messages(self, key, sys_prompt= None, user_message= None, user_input= None):
        # Method to help bundle and store messages to user and chat model

        if key.lower() not in self.keys:
            raise ValueError("Key is not valid. Check class definition.")

        msg_bundle = {}

        if sys_prompt is not None:
            msg_bundle['sys_prompt'] = sys_prompt
        if user_message is not None:
            msg_bundle['user_message'] = user_message
        if user_input is not None:
            msg_bundle['user_input'] = user_input
        
        # Update messages in dictionary
        self.ptc[key.lower()] = msg_bundle
 

class Lexi():
    # This class is capable of managing the whole communication with a chat model integrating external functions, access to real-time data 
    # and NLP capabilities.

    #Constants
    SUCCESSFUL = 1
    WITH_ERRORS = -1

    def __init__(self, model: str = "gpt-3.5-turbo"):

        self.model = model
        self.session_id = self.get_sesid()
        self.cmds_catalog = {}
        self.nid_index = {}
        self.nid_generator = itertools.count(start=1)
        self.indent = 4
        self.response = None
        self.log_file_name = "data/log_data.json"
        self.pinned_pids = []
        self.unfinished_thread = False
        self.generated_prompt = None

        # Change prompt when chatting:
        self.lexi_prompt= "Lexi_>:"
        #Adjust temperature for chat model response
        self.temperature = 0.8 

        # Chat model prompt Components
        self.header_prompt = None
        self.expected_response = None
        self.res_available = []
        self.attached_data = None
        self._last_user_inputs = []

        # Define conversation model
        self.conversation =  {
                "model": f"{self.model}",
                "messages": []
            }

        # Load system variable with API KEY 
        api_key = os.environ.get("MY_API_KEY")
        if api_key:
            #print("API key found!")
            openai.api_key = api_key
        else:
            print("API key not found.")
        
        self.append_basic_IO()

    def append_basic_IO(self):

        # Append internal basic I/O methods / protocols
        search_engine = LexiExternalCommand(func=self.bing_search, 
                                            append_mode= LexiExternalCommand.FIX, 
                                            printer=self.bing_search_printer,
                                            show_return_to_user = True
                                            )
        
        search_engine.update_messages(key='Error',
                                    sys_prompt="There was an error executing the Bing search. Please check the parameters and try again.",
                                    user_message = "I'm sorry, but I wasn't able to retrieve information from Bing search due to an error. Could you please specify your request further or try again later?"
                                    )
        
        search_engine.update_messages(key='Before', user_message = "Im retrieving information from the Internet right now. Please wait a few moments." )                              
        search_engine.update_messages(key='After',
                                        sys_prompt=("This is the information the user requested (has already been printed on screen). Offer the user additional help on analyzing the information. Do not show this info back again to the user."
                                                "Use 'pid_0002' to read any NID_xxxx to further process the information."
                                                "Only you know about the NIDs ids, user has no knowledge. Do not mention NIDs to the user."),
                                        user_message = "I've found this related information:",
                                        user_input = "Please let me know if I can provide with further assistance on your request."
                                    )
        self.append_command(search_engine)

        extract_url = LexiExternalCommand(self.extract_article, 
                                        append_mode= LexiExternalCommand.FIX,
                                        show_return_to_user = True
                                        )

        extract_url.update_messages(key='Error',
                                    sys_prompt= "There was an error executing the Bing search. Please check the parameters and try again.",
                                    user_message = "I'm sorry, but I wasn't able to retrieve information from Bing search due to an error. Could you please specify your request further or try again later?"
                                    )
        
        extract_url.update_messages(key='Before', user_message = "Im extracting the contents of the article right now, just a moment." )                               
        extract_url.update_messages(key='After',
                                        sys_prompt= "Information extracted from NID is being included in the attached data. Use it to assist user request. Protocol for function finishes here.",
                                        user_message = "Here's an extract from the article:",
                                    )
        self.append_command(extract_url)
  
    def append_command(self, commandObject: LexiExternalCommand) -> bool:
        # Append command to catalog
        self.cmds_catalog[commandObject.pid_id] = commandObject

        # Check the append mode to pin the pid to the catalog:
        if commandObject.append_mode is LexiExternalCommand.FIX:
            self.pinned_pids.append(commandObject.pid_id)

    def get_sesid(self):
        sesid_file = 'data/sesid.txt'
        
        # Try to read the existing SESID from the file
        try:
            with open(sesid_file, 'r') as file:
                sesid = int(file.read().strip())
        except FileNotFoundError:
            # If the file does not exist, start with SESID 1
            sesid = 1
        except ValueError:
            # If the value is not an integer, start with SESID 1
            print("The value in sesid.txt is not an integer.")
            sesid = 1

        # Increment SESID
        incremented_value = sesid + 1

        # Write the incremented value back to the file
        with open(sesid_file, 'w') as file:
            file.write(str(incremented_value))

        return incremented_value

    def log_entry(self, log_object) -> bool:
        # Log an entry in the log
        try:
            with open(self.log_file_name, 'a') as file:
                file.write(str(log_object).replace("\n", "") + '\n')
        except IOError as e:
            print(f"An error occurred: {e.strerror}")

    def find_pid_by_name(self, search_name):
        for pid_id, details in self.cmds_catalog.items():
            if 'spec' in details and details['spec'].get('name') == search_name:
                return pid_id
        return None  # Return None if no matching name is found

    def get_pid(self, pid):

        try:
            if pid in self.cmds_catalog:
                new_pid_record = {  'pid':      pid, 
                                    'name':     self.cmds_catalog[pid].specs['name'],
                                    'summ':     self.cmds_catalog[pid].specs['summ'],
                                    'params':   self.cmds_catalog[pid].specs['params']
                    }
            return new_pid_record
        except Exception:
            return None

    def read_local_table(self, data):
        return json.dumps({"table": data}, indent=4)

    def unwrap_url(self, redirected_url: str) -> str:

        parsed_url = urlparse(redirected_url)
        
        # For Bing's URL redirection
        if "bing.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            original_url_encoded = query_params.get('url', [None])[0]
            if original_url_encoded:
                return unquote(original_url_encoded)

        # For Google's URL redirection
        elif "google.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            original_url_encoded = query_params.get('q', [None])[0]
            if original_url_encoded:
                return unquote(original_url_encoded)

        # If no unwrapping pattern is recognized, return the original URL
        return redirected_url 

    def map_news_to_summary(self, news_item):
        global nid_generator  # Use the global nid_generator
        nid = f'nid_{next(self.nid_generator):04}'  # Generates 'nid_XXXX' where XXXX is an incremented value
        title = news_item.get('title', '')
        summary = news_item.get('summary', '')
        url = next((link['href'] for link in news_item.get('links', []) if link.get('rel') == 'alternate'), None)
        
          
        # Internal update of NID to restore the url if needed.
        self.nid_index.update({ nid: {
                                    'title': title,
                                    'summary': summary,
                                    'url': self.unwrap_url(url)
                                    }}
                            )

        return {
            nid: {
                'title': title,
                'summary': summary
            }
        }

    def read_rss(self, rss_url: str, keywords: List[str] = None, req_head: dict = None) -> str:
        # Parsing RSS feeds into JSON
        feed = feedparser.parse(rss_url, request_headers=req_head)
        entries = feed.entries

        # Optionally filter by keywords
        nid_dict = {}
        for entry in entries:
            if not keywords or any(kw.lower() in entry.title.lower() for kw in keywords):
                nid_dict.update(self.map_news_to_summary(entry))

        return json.dumps(nid_dict, indent=4)
    
    def bing_search(self, keywords: List[str]) -> str:
        # KEYS: how who when what which where what since know what's news check
        # SUMM: get real-time information from Bing search engine
        # PTCL: 101

        headers = {
        'Accept-Language': 'en-US,en;q=0.8'
    }
        search = None
        #Treating the keywords in different ways depending the input format
        if isinstance(keywords, str):
            search = "%20".join(keywords.split())

        elif isinstance(keywords, list):
            keywords = [k.replace(" ", "%20") for k in keywords]
            search = "%20".join(keywords)

        if search:
            try:
                rss_url = f"https://www.bing.com/news/search?q={search}&format=RSS"

                # Try to get results in english
                return self.read_rss(rss_url = rss_url, req_head = headers)
            except Exception as e:
                raise ValueError("Problems retrieving informatin from Bing.")
    
    def bing_search_printer(self, json_data) ->str:
        # Function to print title and summary
        output = "\n"
        try:
            news_data = json.loads(json_data)
            for news_id, news_content in news_data.items():
                title = news_content.get('title')
                summary = news_content.get('summary')
                output += f"    -Title: {title}\n     -Summary: {summary}\n\n"
            
            return output
        except json.JSONDecodeError as e:
            raise ValueError(e)

    def extract_article(self, nid: str, url:str = None) -> str:
        # KEYS: nid extract read article url 
        # SUMM: retrieves the text data from a nid_XXXX with article details, gives more detailed information.
        # PTCL: 102

        if url is None:
            if nid not in self.nid_index:
                raise ValueError(f"{nid} not found.")
            #Retreieve the stored URL
            url = self.nid_index[nid]['url']

        article = Article(url)
        article.download()
        article.parse()

        article_data = {
            'nid': nid,
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': article.publish_date.isoformat() if article.publish_date else None,
            'top_image': article.top_image,
            'movies': article.movies,
            'keywords': article.keywords,
            'summary': article.summary
        }

        return json.dumps(article_data, indent=4)
    
    def process_outbound_json(self, generated_prompt: str, reset: bool = False) -> bool:
        # Process the json data and sends it to the chat model

        # If reset is set True, previous messages of the conversation are erased to save reduce Token limitation
        if reset:
            # Define conversation model
            self.conversation =  {
                    "model": f"{self.model}",
                    "messages": []
                }
            
        # Update the conversation messages:
        generated_prompt_dict = json.loads(generated_prompt)
        self.conversation['messages'].extend(generated_prompt_dict)

        # Record output message in log before sending:
        self.log_entry(generated_prompt_dict)
     
        try:
            # Use the API to get a response:
            chat_response = openai.ChatCompletion.create(
                                                        model=self.model,
                                                        messages= self.conversation['messages'],
                                                        temperature = self.temperature
                                                        )
            
            self.response = chat_response['choices'][0]['message']['content']

        except Exception as e:
            raise ValueError("Problems conneting to openAI. ", e)

    def new_user_input(self, user_input: str, send_message: bool = True) -> str:

        # Determine how to start setting the context
        # Check if Lexi is not waiting for further user input:
        if self.unfinished_thread is False:
            
            # New thread, create a new context:   
            self.header_prompt = lexi_dictionary['new_user_message']

            self.res_available = []
            # Add the pinned pids (always available to chat model)
            for pid in self.pinned_pids:
                new_resource = self.get_pid(pid)
                self.res_available.append(new_resource)
            
            # Define the expected response if are resources available:
            self.expected_response = lexi_dictionary['resources_detected']
        
        else:
            # Close the thread / use previous context:
            self.unfinished_thread = False
        
        # Update which methods / protocols are applicable 
            for pid in self.cmds_catalog.keys():
                # Check if any word in the user input is among the method key words
                if pid not in self.res_available and any([True if word.lower() in self.cmds_catalog[pid].specs['keys'] else False for word in user_input.split()]):
                    new_resource = self.get_pid(pid)
                    self.res_available.append(new_resource)

        # Complete the next message with the user input, use previous context:
        generated_prompt = self.new_json_template_out(header_prompt = self.header_prompt, 
                                                            user_input = user_input,    # Add user input 
                                                            available_resources = self.res_available, 
                                                            expected_response = self.expected_response, 
                                                            attach_data = self.attached_data)

        # Clear all the internal states just in case:
        self.header_prompt = None
        self.expected_response = None
        self.res_available = []
        self.attached_data = None

        # Execute the outbound process:
        if send_message:
            self.process_outbound_json(generated_prompt, reset= True)
   
    def new_system_action(self, system_level_prompt: str = None, pids: List[str] = None, data = None, pid = None, return_status = None) -> str:
        # Requests an action from Lexi (messsage to user, further interaction with chat model).

        # Convert to str (just in case it was a dictionary or list)
        self.header_prompt = str(system_level_prompt) if system_level_prompt else ""

        # Options available for the model
        self.res_available = []
        # Add the specs of the pids
        if pids:
            for pid in pids:
                self.res_available.extend(self.get_pid(pid))

        self.expected_response = None
        # Check return status of last command executed:
        if return_status is Lexi.SUCCESSFUL:

            # Recover PID protocol
            ptcl = self.cmds_catalog[pid].ptc

            # Check if protocol requires a message to the user
            if ptcl['after'] is not None and 'user_message' in ptcl['after']:
                print(ptcl['after']['user_message'])
            # Also check if return from funtion is needed:
            try:
                if ptcl['show_return_to_user'] is True:
                    print(self.format_user_response(pid, data))
            except Exception as e:
                pass

            # Check for nexts steps
            if ptcl['after'] is not None and 'sys_prompt' in ptcl['after']:

                try:
                    # Check for additional resources to include in the prompt
                    if ptcl['next'] is not None and 'call' in ptcl['next']:
                        func_name = ptcl['next']['call']
                        # Associated pids in protocol
                        protocol_pid = self.find_pid_by_name(func_name)
                        self.res_available.append(self.get_pid(protocol_pid))
                except Exception:
                    pass

                # Attach data from previous call
                self.attached_data = data

                # Check if an additional input from the user is needed, this will end the current prompt processing and
                # let an unfinished thread open (True), so in the next call to "new_user_input" the Class will know that
                # the next prompt for the chat model is almost ready, just missing a user input.

                if 'user_input' in ptcl['after']:
                    print(self.lexi_prompt, ptcl['after']['user_input'])
                    self.unfinished_thread = True

                # Specify the request coming from the protocol
                self.header_prompt = ptcl['after']['sys_prompt']
                self.expected_response = lexi_dictionary['new_user_message']                 

            elif return_status is Lexi.WITH_ERRORS:
                # Manage errors messages:
                if 'error' in ptcl:
                    self.header_prompt += ptcl['error']['sys_prompt']
                    self.attached_data= ""

                # Check if there is a message to show to the user
                if ptcl['error'] is not None and 'user_message' in ptcl['error']:
                    print(ptcl['error']['user_message'])

        # Send a message back to the chat model if the protocol requires it:
        if self.expected_response:
            generated_prompt = self.new_json_template_out(header_prompt = self.header_prompt, 
                                                            # user_input = add_user_input, No user input needed 
                                                            available_resources = self.res_available, 
                                                            expected_response = self.expected_response, 
                                                            attach_data = self.attached_data)

            if self.unfinished_thread is False:
                # Execute the outbound process:
                self.process_outbound_json(generated_prompt)

                # Callback for Lexi to process the chat model response:
                self.process_inbound()
            
            else:
                # Keep a copy of the generated prompt:
                self.generated_prompt = generated_prompt

    def format_user_response(self, pid: str, data) -> str:
        # Format external commands return values to show to user, can be used as a tailored output solution
        
        # If ptcl is not recognized
        if pid not in self.cmds_catalog or self.cmds_catalog[pid].printer == None:
            try:
                # Check if data is a string that can be loaded as JSON
                if isinstance(data, str):
                    json_object = json.loads(data)
                elif isinstance(data, dict):
                    # It's already a dictionary
                    json_object = data
                else:
                    # If formatting does not work, show data dump:
                    return data

                # If data is a valid JSON object, pretty-print it with default json printer included in Lexi Class:
                return self.build_pretty_json_string(json_object)

            # If formatting does not work, show data dump:
            except Exception as e:
                return data
        else:
            # The command has a defined printer to format the output of the process
            printer = self.cmds_catalog[pid].printer
            try:
                return printer(data)
            
            # If formatting does not work, show data dump:
            except Exception as e:
                return data

    def build_pretty_json_string(self, data, indent=0):
        result_lines = []

        # Determine the type of the data
        if isinstance(data, dict):
            for key, value in data.items():
                result_lines.append('  ' * indent + str(key) + ':')
                result_lines.append(self.build_pretty_json_string(value, indent + 1))
        elif isinstance(data, list):
            for item in data:
                result_lines.append(self.build_pretty_json_string(item, indent))
        else:
            # This is a leaf node (neither dict nor list)
            result_lines.append('  ' * indent + str(data))

        return '\n'.join(result_lines)

    def process_inbound(self) -> bool:
        # Process

        if not self.response:
            return False
        else:
            # First log the response from the chat model
            self.log_entry(self.response)
            # Parse the JSON-formatted string
            try:
                #
                parsed_content = json.loads(self.response)
                # Access the fields in parsed_content
                user_level_response = parsed_content.get("user_level")
                system_level_response = parsed_content.get("system_level")

                # If you need to take action based on the system_level_response
                selected_pid = system_level_response.get("selected_pid")
                parameters = system_level_response.get("parameters")
                status = system_level_response.get("status")
                error_message = system_level_response.get("error_message")

                # Show message to user if any:
                if user_level_response:
                    print(self.lexi_prompt, user_level_response['resp_text'])

                # If an action is required (a pid was selected by the chat model), trigger the corresponding protocol:
                try:
                    if selected_pid in self.cmds_catalog:
                        self.execute_external_command(selected_pid, parameters)
                except Exception as e:
                    print("Error calling protocol:", selected_pid, e)
                
            except Exception as e:
                    print("Error decoding:", self.response, "\n", e)

            # once processed, empty the last response from the chat model.
            self.response = None

        return True

    def new_json_template_out(self, header_prompt = None, available_resources = None, expected_response = None, user_input = None, attach_data = None):
            
        # Construct system message with nested JSON structure
        #         
        system_message = {
            "role": "system",
            "content": ".".join([lexi_dictionary['new_user_message'], lexi_dictionary['json_format']])
                    }

        usr_content = {
                    "system_level": {
                        "session": self.session_id,
                        "header": str(header_prompt),
                        "expected_action": str(expected_response),
                        "pids_available": str(available_resources),          
                    },
                    "user_level" : {
                        "user_text" :str(user_input),
                        "message_history" : str(self._last_user_inputs),
                        "attached_data" : str(attach_data)
                    }
        }

        # Construct user message
        user_message = {
            "role": "user",
            "content": json.dumps(usr_content)
        }
        
        # Combine both messages into a list
        messages = [system_message, user_message]

        # Update message history (to add context in the next interaction):
        # Append the new input to the list
        self._last_user_inputs.append(user_input)
        # Keep only the last three inputs
        self._last_user_inputs = self._last_user_inputs[-3:]

        # Convert the structure to JSON
        return json.dumps(messages, indent=self.indent)
   
    def parse_json_values(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self.parse_json_values(value)
        elif isinstance(data, str):
            try:
                data = data.replace("'", '"')
                data = json.loads(data)
                # If the result is a dictionary, recurse into it
                if isinstance(data, dict):
                    data = self.parse_json_values(data)
            except json.JSONDecodeError:
                # The string was not a JSON, so we just return it unchanged
                pass
        return data

    def execute_external_command(self, pid: str, params: dict) -> bool:
        # Calls a external command (PID) and addresses calls for a new system action

        # Retrieve the function to be called
        function = self.cmds_catalog[pid].call
        
        # Check if the protocol includes a message to the user before executing the command:
        try:
            user_msg = self.cmds_catalog[pid].ptc['before']
            if 'user_message' in user_msg:
                print(self.lexi_prompt, user_msg['user_message'])

        except Exception:
            pass

        # Check if function exists and is callable
        if callable(function):
            # Call the function (PID) with the unpacked parameters:
            try:
                parameters = self.parse_json_values(params)

                if isinstance(parameters, dict):
                    call_return_value = function(**parameters)

                elif isinstance(parameters, str):
                    call_return_value = function(parameters)

                elif parameters is None:
                    call_return_value()

                self.log_entry(f"Function {function.__name__} excecuted with params:{params}")

                # After execution, a new system action is needed to 
                try:
                    # Create prompt for chat model following the protocol
                    self.new_system_action(pid= pid, return_status= Lexi.SUCCESSFUL , data= call_return_value)
                except Exception as e:
                    msg = f"Error ocurred when creating system action for {function.__name__}: {e}."
                    print(msg)
                    self.log_entry(msg)

                return True
            
            # When errors occur:
            except Exception as e:
                msg = f"Error ocurred when calling function {function.__name__}: {e}"
                print(msg)
                self.log_entry(msg)

                # Request system action followring the protocol after an error.
                self.new_system_action(pid= pid, return_status = Lexi.WITH_ERRORS )

                return False
        else:
            msg = f"Function {function.__name__} not found or is not callable."
            print(msg)
            self.log_entry(msg)
            return False

    def mini_chat_session(self):
        # Start mini chat
        user_text = input("You:")
        while user_text != "exit":

            self.new_user_input(user_text)
            self.process_inbound()

            user_text = input("You:")

       
if __name__ == "__main__":

    lexi = Lexi()
    lexi.mini_chat_session()
    
