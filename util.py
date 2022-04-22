


class Button:
    def __init__(self, img, hitbox, tag):
        self.img = img
        self.hitbox = hitbox
        self.tag = tag
    
    def hit(self, coord):
        hb = self.hitbox
        x = coord[0]
        y = coord[1]
        top = hb[0]
        left = hb[1]
        bottom = hb[2]
        right = hb[3]

        if x < right and x > left and y < bottom and y > top:
            return True
