import json
motion_dict =   [
                    {
                        "name": "sit", 
                        "duration": 20, 
                        "display": "sit", 
                        "desc": "Please sit"
                    },
                    {
                        "name": "stand", 
                        "duration": 20, 
                        "display": "stand", 
                        "desc": "Please stand"
                    },
                    {
                        "name": "walk", 
                        "duration": 20, 
                        "display": "walk", 
                        "desc": "Please walk on a firm and flat ground"
                    },
                    {
                        "name": "upstairs", 
                        "duration": 20, 
                        "display": "upstairs", 
                        "desc": "Please go upstairs at a steady and normal speed"
                    },
                    {
                        "name": "downstairs", 
                        "duration": 20, 
                        "display": "downstairs", 
                        "desc": "Please go downstairs at a steady and normal speed"
                    },
                    {
                        "name": "run", 
                        "duration": 20, 
                        "display": "run", 
                        "desc": "Please run on a firm and flat ground"
                    }
                ]
with open("config/motions.json", "w") as f:
    json.dump(motion_dict, f)