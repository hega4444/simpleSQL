
import inspect
import xmltodict
import feedparser
from typing import List
import requests
from urllib.parse import urlparse, parse_qs, unquote
from newspaper import Article
import xml.etree.ElementTree as ET


class Lexi():

    def __init__(self, prompt_init = None) -> None:
        self.prompt_init = prompt_init
        self.cmds_catalog = None
        self.cmds_caller = None
        self.pid_counter = 1
        self.append_function(self.xml_check_rss_sources)
        self.append_function(self.xml_from_table)
        self.append_function(self.xml_read_rss)
        self.nid_index = {}
        self.nid_counter = 1

    def append_function(self, func,  prompt = "null"):
        #Appends a function to the command catalog
        sig = inspect.signature(func)
        
        # Create id for PID
        pid_id = "pid_" + str(self.pid_counter).zfill(3)
        self.pid_counter +=1

        #Function name
        pid_name = func.__name__

        # Extract return annotation if available
        return_annotation = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else "null"

        # Check for annotations
        det_annotation = func.__annotations__

        # Determine which parameters are optional or mandatory
        params = {}

        for name, param in sig.parameters.items():
            new_param = {}
            if param.default == inspect.Parameter.empty:
                new_param['req'] = 'y'
            else:
                new_param['req'] = 'n'
            if name in det_annotation:
                new_param['type'] = self.text_type(str(det_annotation[name]))
            params[name] = new_param

        # Chek for comments on the source code of method
        source_lines = inspect.getsource(func).split('\n')
        comments = ''.join([line.strip() for line in source_lines if line.strip().startswith("#")])

        new_function =  {'name':  pid_name,
                        'prompt': prompt,
                        'parms': params,
                        'return': return_annotation,
                        'cmmnt': comments
                        }
        
        # Add function to the catalog
        if self.cmds_catalog is not None:
            self.cmds_catalog[pid_id] = new_function
            self.cmds_caller[pid_id] = func
        else:
            self.cmds_catalog = {}
            self.cmds_caller = {}
            self.cmds_catalog[pid_id] = new_function
            self.cmds_caller[pid_id] = func

    def stringify_types(self, data):
        #Convert type objects in a nested dictionary to their string representations.

        if isinstance(data, dict):
            return {k: self.stringify_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.stringify_types(item) for item in data]
        elif isinstance(data, type):
            return self.text_type(str(data))
        else:
            return data
    
    def text_type(self, text):
        if 'int' in text:
            return 'int'
        elif 'str' in text:
            return 'str'
        elif 'float' in text:
            return 'float'
        elif 'bool' in text:
            return 'bool'

    def xml_create_catalog(self, pretty = True):
        # Wrap the original dictionary inside another dictionary with a single root key
        
        data_with_root = {'__catalog__': self.cmds_catalog}
        dict_pre_converted = self.stringify_types(data_with_root)
        xml_string = xmltodict.unparse(dict_pre_converted, pretty = pretty)

        with open("prompt_data/__catalog__.prompt", "w") as file:
            file.writelines(xml_string)
        
        return xml_string
    
    def xml_from_table(self, data, pretty = True):
        
        xml_table = {'table': data}

        return xmltodict.unparse(xml_table, pretty = pretty)

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

    def xml_read_rss(self, rss_url : str, keywords: List[str] = None, req_head: dict = None, pretty: bool = True) -> str:
        # Read rss feed and create xml file with digest

        feed = feedparser.parse(rss_url, request_headers = req_head)
        news = feed['entries']

        dic_news = {}
        for new in news:

            if not keywords or all ([True if key.lower() in new['title'].lower() or \
                                     key.lower() in new['summary'].lower() else False for key in keywords]):

                # Add entry to dict for xml generarion
                sum = new['summary'][:200] if len(new['summary'])>200 else new['summary']
                # Create news identifier
                nid = "nid_" + str(self.nid_counter).zfill(4)
                dic_news[nid] = {'title': new['title'], 'sum': sum }
                
                # Keep an internal entry to gather the URL if needed
                self.nid_index[nid] = {'title': new['title'], 'url': self.unwrap_url(new['link'])}
                #Update id counter
                self.nid_counter +=1 
        
        # Update nid counter (nid is the index of the rss feed sent to CGPT, to keep track of the valid news id)
        data_news = {'news' : dic_news}
        xml_news =  xmltodict.unparse(data_news, pretty = pretty)

        return xml_news
    
    def xml_check_rss_sources(self, keywords: List[str] = None, limit = 100, pretty: bool = True) -> str:
        # Search news containing "keywords", "limit": max number of results
        # Checks all the rss channels listed in "__sources__.prompt"

        with open("prompt_data/__sources__.prompt", "r", encoding='utf-8') as file:

            # Parse the XML content into a dictionary
            xml_content = file.read()
            dict_content = xmltodict.parse(xml_content)

            # Accessing elements in the dictionary
            rss_feeds = dict_content['rss_feeds']['url']

            dic_news = {}
            id = 1

            for feed in rss_feeds:

                dic_rss = xmltodict.parse(self.xml_from_rss(rss_url=feed, keywords= keywords, pretty=pretty))

                if isinstance(dic_rss['news'], dict):
                    for new in dic_rss['news'].values():
                        dic_news["nid_" + str(id).zfill(4)] = new
                        id += 1

                        if id > limit:
                            break

                if id > limit:
                    break

            data_news = {'news' : dic_news}
            self.nid_counter +=1
            xml_news =  xmltodict.unparse(data_news, pretty = pretty)

            return xml_news

    def xml_bing_search(self, keywords: List[str], pretty: bool = True) -> str:
        # Retrieves a XML feed based on "keywords"

        headers = {
        'Accept-Language': 'en-US,en;q=0.8'
    }

        search = '%20'.join(keywords)
        rss_url = f"https://www.bing.com/news/search?q={search}&format=RSS"

        # Try to get results in english
        return self.xml_read_rss(rss_url = rss_url, req_head = headers,  pretty= pretty)
    
    def extract_article(self, url: str, nid: str) -> str:
       
        # Extract text content from the article
        article = Article(url)
        article.download()
        article.parse()

        # Build XML structure with the relevant info
        root = ET.Element(nid)
        content = ET.SubElement(root, "content")
        content.text = article.text 

        xml_string = ET.tostring(root, encoding="unicode")
        xml_string =  '\n'.join([line for line in xml_string.splitlines() if line.strip() != ""])

        return xml_string

    def xml_from_nid(self, nid: str) -> str:
        #Retuns a XML structure based on its indexed NID (news identifier).

        if nid in self.nid_index:
            return self.extract_article(self.nid_index[nid]['url'], nid)

    def execute_command(self, pid: str, *args, **kwargs):
        #Execute a command listed in the __catalog__.prompt

        if pid in self.cmds_caller:
            function = self.cmds_caller[pid]
        
        # Try calling the function passing the 
        try:
            ret = function(*args, **kwargs)
            return ret
        except Exception as e:
            print(f"Problems executing {pid}.") 

    def process_inboud_xml(self, xml_data: str) -> bool:

        # Parse the XML
        root = ET.fromstring(xml_data)

        # Iterate through each action
        for act in root.findall('acts/act'):
            level = act.get('lvl')
            if level == 'sys':
                action_type = act.find('tp').text
                pid = act.find('pid').text
                params = [prm.text for prm in act.findall('prms/prm')]
                status = act.find('st').text
                print(f"System Action: Type={action_type}, PID={pid}, Params={params}, Status={status}")

                self.execute_command("pid_"+pid, *params)

            elif level == 'usr':
                text = act.find('txt').text
                print(f"Text for user: {text}")
       
if __name__ == "__main__":

    def test_function(val_a: int, val_b: str, another: bool = None ) -> int:
        #This is a test function.
        print("adjusting lights", val_a, val_b)
        return 1

    lexi = Lexi()

    lexi.append_function(test_function)


    lexi.xml_create_catalog(pretty=True)

    #print(lexi.xml_check_rss_sources(keywords=['Israel', 'war'], limit=10))

    #print(lexi.xml_from_rss(rss_url='https://www.google.com/alerts/feeds/09504428241442793234/3573248333781749813'))
    #print(lexi.xml_from_rss(rss_url='https://www.bing.com/news/search?q=friends%20tv&format=RSS'))

    print(lexi.xml_bing_search(keywords=['Covid', 'germany']))

    #print(lexi.xml_from_nid('nid_0009'))

    #lexi.execute_command('pid_004', 1, 45) #execute a command by pid id

    xml_data = """
<rsp>
    <acts>
        <act lvl="sys">
            <tp>exec</tp>
            <pid>004</pid>
            <prms>
                <prm>bedroom</prm>
                <prm>90</prm>
            </prms>
            <st>pending</st>
        </act>
        <act lvl="usr">
            <txt>I've adjusted the bedroom light to 90%.</txt>
        </act>
    </acts>
</rsp>
"""

    lexi.process_inboud_xml(xml_data)