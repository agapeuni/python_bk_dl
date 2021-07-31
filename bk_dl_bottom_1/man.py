class Man:
    def __init__(self, name):
        self.name = name
        print("__init__")
        
    def hello(self):
        print("Hello " + self.name)
        
    def bye(self):
        print("Bye " + self.name)
        
man = Man("John")
man.hello()
man.bye()
        