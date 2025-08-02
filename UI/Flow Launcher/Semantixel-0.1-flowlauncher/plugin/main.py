from flowlauncher import FlowLauncher
import webbrowser
import urllib.request
import json
from PIL import Image


class Semantixel(FlowLauncher):
    def send_query(self, query):
        # embedding text search (search in OCRed image content)
        if query.startswith("|"):
            # remove | from the query
            query = query.lstrip("|")
            query = query.lstrip()
            data = json.dumps({"query": query}).encode()
            req = urllib.request.Request(
                "http://localhost:23107/ebmed_text",
                data=data,
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    return None
        # (clip image search)
        if query.startswith("#"):
            # remove # from the query
            query = query.lstrip("#")
            query = query.lstrip()
            data = json.dumps({"query": query}).encode()
            req = urllib.request.Request(
                "http://localhost:23107/clip_image",
                data=data,
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    return None

        # else (clip text search)
        else:
            data = json.dumps({"query": query}).encode()
            req = urllib.request.Request(
                "http://localhost:23107/clip_text",
                data=data,
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    return None

    def query(self, query):
        results = self.send_query(query)
        results_dict = []
        i = 100
        for result in results:
            results_dict.append(
                {
                    "title": result.split("\\")[-1],
                    "subtitle": result,
                    "icoPath": result,
                    "jsonRPCAction": {
                        "method": "open_path",
                        "parameters": [result],
                    },
                    "score": i,
                    "context": result,
                }
            )
            i -= 1
        return results_dict

    def open_path(self, path):
        path = path.replace("\\", "\\\\")
        Image.open(path).show()
