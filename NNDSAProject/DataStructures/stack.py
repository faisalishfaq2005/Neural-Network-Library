class Stack:
    def __init__(self):
        self.stack=[]
    
    def push(self,data):
        self.stack.append(data)

    def is_empty(self):
        if len(self.stack)==0:
            return True
        else:
            return False
        

    def pop(self):
        if self.is_empty():
            print("stack empty cannot pop")
        else:
            data=self.stack.pop()
            return data
    
    def peek(self):
        data=self.stack[-1]
        return data
    
