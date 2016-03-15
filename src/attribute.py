class Attribute(object):
    def __init__(self, attrName, attrValues):
        self.attrName = attrName
        self.attrValues = attrValues

    def __str__(self):
        return "Attribute Name: {} || AttributeValue(s): {}\n".format(self.attrName, self.attrValues)

    def __repr__(self):
        return str(self)