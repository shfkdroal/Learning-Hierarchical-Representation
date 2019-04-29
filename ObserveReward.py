
import sys
import turtle
import os

import tensorflow as tf
import random
import numpy as np
import graphic
import matplotlib.pyplot as plt


#Game Param
w = 60
agentSpeed = 60
Num_product_category = 4
Num_product_for_each = 2
Num_desire = 2
Maximum_inventory = 10
Maximum_object_recognition = 45 #200

StateDim = 2 + (Maximum_inventory + 1) + Maximum_object_recognition
ActionDim = 4
State_and_Action_Dim = StateDim + ActionDim
Num_col = 2 * StateDim + 1
BatchSize = 1

#Gloabal Variable 1
AgentCoord = [0, 0]
QualityPopupList = []

currentState = np.zeros([BatchSize, StateDim, 1, 1], dtype=float)

#class definition
class Product:
    def __init__(self, x=0, y=0, k=0, idx=0, qf=0):
        self.kind = k
        self.idx_within_kind = idx
        self.quality_factor = qf
        self.xCoord = x
        self.yCoord = y

    def getProductInfo(self):
        print("kind: {0}, idx: {1}, quality: {2}".format(self.kind, self.idx_within_kind, self.quality_factor))

    def getProducCoord(self):
        print("y: {0}, y: {1}".format(self.xCoord, self.yCoord))


class State:
    def __init__(self, x=[], internal=[], num_product=0, coord=[0, 0]):
        self.Num_Prod = num_product

        self.Num_have = 0

        self.current_coord = coord
        self.External_state_Buffer = x
        self.Internal_state_Buffer = internal

        for ii in range(Maximum_object_recognition):
            self.External_state_Buffer.append(-1)

        for ii in range(Maximum_inventory+1):
            self.Internal_state_Buffer.append(-1)

        self.Internal_state_Buffer[Maximum_inventory] = random.randint(0, Num_desire - 1)

    def init(self, x=[], internal=[], num_product=0, coord=[0, 0]):
        self.Num_Prod = num_product

        self.Num_have = 0

        self.current_coord = coord
        self.External_state_Buffer = x
        self.Internal_state_Buffer = internal

        for ii in range(Maximum_object_recognition):
            self.External_state_Buffer.append(-1)

        for ii in range(Maximum_inventory+1):
            self.Internal_state_Buffer.append(-1)

        self.Internal_state_Buffer[Maximum_inventory] = random.randint(0, Num_desire - 1)

    """
        self.kind = k
        self.idx_within_kind = idx
        self.quality_factor = qf

        self.xCoord = x
        self.yCoord = y

    """

    #State Initialization
    def Update_External_State_0(self, productList):
        self.current_coord = [0, 0]
        for np in range(len(productList)):
            v = productList[np]
            self.External_state_Buffer[5*np] = v.kind
            self.External_state_Buffer[5*np+1] = v.idx_within_kind
            self.External_state_Buffer[5*np+2] = -1
            self.External_state_Buffer[5*np+3] = v.xCoord
            self.External_state_Buffer[5*np+4] = v.yCoord
            self.Num_Prod += 1

        #self.record_state()


    #Nurture Information when colide
    def Update_External_State_1(self, productList, colideIdx):

        #print(colideIdx)

        if colideIdx == -2:
            #the product was already handled
            for np in range(len(productList)):
                v = productList[np]
                self.External_state_Buffer[5 * np] = v.kind
                self.External_state_Buffer[5 * np + 1] = v.idx_within_kind
                self.External_state_Buffer[5 * np + 2] = -1
                self.External_state_Buffer[5 * np + 3] = v.xCoord
                self.External_state_Buffer[5 * np + 4] = v.yCoord

            self.External_state_Buffer[5 * (self.Num_Prod - 1)] = -1
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 1] = -1
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 2] = -1
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 3] = -1
            self.External_state_Buffer[5 * (self.Num_Prod - 1) + 4] = -1

            self.Num_Prod -= 1

        elif colideIdx != -1:
            #print(self.External_state_Buffer)
            self.External_state_Buffer[5 * colideIdx + 2] = productList[colideIdx].quality_factor
            note = turtle.Turtle()
            note.penup()
            note.setposition(w*productList[colideIdx].xCoord - 270, w*productList[colideIdx].yCoord - 300)
            note.color("white")
            #print(len(productList))
            #print(colideIdx)
            #print(len(ProdCoordList))
            #print(len(ProductList))
            note.write("{}".format(productList[colideIdx].quality_factor))
            QualityPopupList.append(note)

        #print(self.External_state_Buffer)
        #numbers = [num + 1 for num in self.External_state_Buffer]
        #print(self.External_state_Buffer + self.current_coord + self.Internal_state_Buffer)
        self.current_coord = AgentCoord
        self.record_state()
        return

    def record_state(self):
        global currentState
        """
                self.Num_Prod = num_product

                self.current_coord = coord
                self.External_state_Buffer = x
                self.Internal_state_Buffer = internal

                Plus 1 onto every elements

        """

        #f = open("img.bmp", 'a+b')
        currentState = np.array((self.current_coord + [a + 1 for a in self.External_state_Buffer] +
                                 [b + 1 for b in self.Internal_state_Buffer]), dtype=float).reshape(1, StateDim, 1, 1)

        return


def record_action(action_took):
    return

##########################


state = State()
wn = turtle.Screen()
wn.bgcolor("black")
wn.title("Chef")


##########################

borderpen = turtle.Turtle()
borderpen.speed(0)
borderpen.color("white")
borderpen.penup()
borderpen.setposition(-300, -300)
borderpen.pendown()
borderpen.pensize(3)

for side in range(4):
    borderpen.fd(600)
    borderpen.lt(90)

borderpen.pensize(0.5)
WxCoord = -300
WyCoord = -300
for i in range(int(2*300/w)):
    WxCoord = WxCoord + w
    borderpen.penup()
    borderpen.setposition(WxCoord, -300)
    borderpen.setheading(90)
    borderpen.pendown()
    borderpen.fd(600)
for i in range(int(2*300/w)):
    WyCoord = WyCoord + w
    borderpen.penup()
    borderpen.setposition(-300, WyCoord)
    borderpen.setheading(0)
    borderpen.pendown()
    borderpen.fd(600)

borderpen.hideturtle()

###########################

agent = turtle.Turtle()
agent.color("pink")
agent.shape("triangle")
agent.penup()
agent.speed(0)
agent.setposition(-300, -300)
agent.setheading(90)

###########################

state_info = turtle.Turtle()

ProductList = []
ProdCoordList = []
CoordList = [-1]*121
Turtle_objectList = [0]*121
#current coord
#mission_code = random.randint(0, 1)

###########################

success_prob = 0.9

#util function (MISC)
def xy_rand_generator():
    x = random.randrange(-300, 300, w)
    y = random.randrange(-300, 300, w)
    #coord = [x, y]
    coord = [int((x+300)/w), int((y+300)/w)]
    if coord not in ProdCoordList:
        ProdCoordList.append(coord)
        return coord
    else:
        coord = xy_rand_generator()
        return coord

def spread_product(num_product, num_kind):

    for k in range(num_kind):
        for p in range(num_product):
            coords = xy_rand_generator()
            single_product = Product(coords[0], coords[1], k, p, random.randint(2, 10))
            # coord, kind, index, quality
            ProductList.append(single_product)

    k = 0
    for v in ProductList:
        prod_rend = turtle.Turtle()
        random.seed(v.kind)
        r = ((100**v.kind+1) % (250 - random.randint(50, 100)))
        g = ((100*(v.kind+2)) % 250)
        b = ((100**v.kind+4) % (250 - random.randint(50, 100)))
        wn.colormode(255)
        prod_rend.color(r, g, b)
        prod_rend.shape("circle")
        prod_rend.penup()
        prod_rend.speed(0)
        prod_rend.setposition(w*v.xCoord - 300, w*v.yCoord - 300)
        CoordList[(11*v.xCoord) + v.yCoord] = k
        #print("cat: {}, coord: ({},{})".format(v.kind, v.xCoord, v.yCoord))
        Turtle_objectList[(11*v.xCoord) + v.yCoord] = prod_rend
        k += 1

    state.Update_External_State_0(ProductList)
    show_Internal_state_on_screen()
    collisionTest()

def collisionTest():
    global AgentCoord
    colideIdx = -1
    X = int((agent.xcor()+300)/w)
    Y = int((agent.ycor()+300)/w)
    AgentCoord = [X, Y]
    k =CoordList[11*AgentCoord[0] + AgentCoord[1]]
    if k != -1:
        colideIdx = k
        #if colideIdx >= len(ProdCoordList):
        newK = 0
        for cd in ProdCoordList:
            if cd[0] == X and cd[1] == Y:
                colideIdx = newK
                CoordList[11 * AgentCoord[0] + AgentCoord[1]] = newK
            newK += 1

    #wn.delay(200)
    state.Update_External_State_1(productList=ProductList, colideIdx=colideIdx)
    return


def show_Internal_state_on_screen():

    state_info.penup()
    state_info.speed(0)
    state_info.color("white")
    state_info.setposition(-300, 350)
    desire = state.Internal_state_Buffer[Maximum_inventory]
    inventories = state.Internal_state_Buffer[0:Maximum_inventory-1]
    infoString = "desire: " + str(desire)
    if desire == 0:
        infoString += " redPurple, gold"
    else:
        infoString += " green, vividPurple"

    state_info.write(infoString, font=("Arial", 10, "normal"))
    state_info.penup()
    state_info.setposition(-100, 350)
    state_info.write(str(inventories), font=("Arial", 10, "normal"))
    print(infoString)
    return

#action definition
def move_left_():
    x = agent.xcor()
    if x > -300:
        x -= agentSpeed
    agent.setx(x)
    agent.setheading(180)
    collisionTest()
def move_right_():
    x = agent.xcor()
    if x < 300:
        x += agentSpeed
    agent.setx(x)
    agent.setheading(0)
    collisionTest()
def move_up_():
    y = agent.ycor()
    if y < 300:
        y += agentSpeed
    agent.sety(y)
    agent.setheading(90)
    collisionTest()
def move_down_():
    y = agent.ycor()
    if y > -300:
        y -= agentSpeed
    agent.sety(y)
    agent.setheading(90)
    collisionTest()

def move_left():
    #wn.delay(20)
    record_action(100)
    t = random.uniform(0, 10)
    if t > 1:
        x = agent.xcor()
        if x > -300:
            x -= agentSpeed
        agent.setx(x)
        agent.setheading(180)
    elif t <= 0.2:
        move_right_()
        return
    elif t <= 0.6:
        move_up_()
        return
    elif t <= 1:
        move_down_()
        return
    collisionTest()
    wn.delay(5)


def move_right():
    #wn.delay(20)
    record_action(200)
    t = random.uniform(0, 10)
    if t > 1:
        x = agent.xcor()
        if x < 300:
            x += agentSpeed
        agent.setx(x)
        agent.setheading(0)
    elif t <= 0.2:
        move_left_()
        return
    elif t <= 0.6:
        move_up_()
        return
    elif t <= 1:
        move_down_()
        return
    collisionTest()
    wn.delay(5)


def move_up():
    #wn.delay(20)
    record_action(300)
    t = random.uniform(0, 10)
    if t > 1:
        y = agent.ycor()
        if y < 300:
            y += agentSpeed
        agent.sety(y)
        agent.setheading(90)
    elif t <= 0.2:
        move_down_()
        return
    elif t <= 0.6:
        move_right_()
        return
    elif t <= 1:
        move_left_()
        return
    collisionTest()
    wn.delay(5)

def move_down():
    #wn.delay(20)
    record_action(400)
    t = random.uniform(0, 10)
    if t > 1:
        y = agent.ycor()
        if y > -300:
            y -= agentSpeed
        agent.sety(y)
        agent.setheading(-90)
    elif t <= 0.2:
        move_up_()
        return
    elif t <= 0.6:
        move_right_()
        return
    elif t <= 1:
        move_left_()
        return
    collisionTest()
    wn.delay(5)


def cook():
    wn.delay(20)
    #for i in range(state.Num_Prod):
    record_action(999)
    global state
    state.Internal_state_Buffer = [0] * (Maximum_inventory + 1)
    state.Internal_state_Buffer[Maximum_inventory] = random.randint(0, Num_desire - 1)
    state.Num_have = 0

    #clear setting
    agent.penup()
    agent.speed(0)
    agent.setposition(-300, -300)
    agent.setheading(90)

    state.init(x=[], internal=[], num_product=0, coord=[0, 0])

    global ProductList
    global ProdCoordList
    ProdCoordList = []
    ProductList = []
    global CoordList
    CoordList = [-1] * 121
    global Turtle_objectList
    global QualityPopupList
    for t in Turtle_objectList:
        if t != 0:
            t.hideturtle()
            t.clear()
    for q in QualityPopupList:
        q.hideturtle()
        q.clear()
    Turtle_objectList = [0] * 121
    QualityPopupList = []

    random.seed(os.times())
    spread_product(Num_product_for_each, Num_product_category)
    state_info.clear()
    show_Internal_state_on_screen()

    return


def get():
    #wn.delay(50)
    record_action(500)
    global ProductList
    global ProdCoordList
    global CoordList
    global Turtle_objectList
    global state
    global AgentCoord

    k = CoordList[11*AgentCoord[0] + AgentCoord[1]]
    #print(CoordList)
    #print("{}, {}  :k:{}".format(AgentCoord[0], AgentCoord[1], k))

    if k == -1:
        return

    #print(len(ProductList))
    #print(k)
    scaled121 = 11*ProdCoordList[k][0] + ProdCoordList[k][1]
    CoordList[scaled121] = -1
    obj = Turtle_objectList[scaled121]
    obj.hideturtle()

    state.Internal_state_Buffer[state.Num_have] = ProductList[k].kind
    if state.Num_have < Maximum_inventory - 1:
        state.Num_have += 1

    del ProductList[k]
    del ProdCoordList[k]
    state_info.clear()
    show_Internal_state_on_screen()

    state.Update_External_State_1(productList=ProductList, colideIdx=-2)
    wn.delay(5)


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    return

tf.reset_default_graph()
set_seed(21)

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/Gail/gail.meta')
print(saver)
saver.restore(sess, tf.train.latest_checkpoint('./model/Gail'))
graph = tf.get_default_graph()
noise = graph.get_tensor_by_name("noise:0") #tensor
is_training = graph.get_tensor_by_name("is_training:0")
feed_sub = graph.get_tensor_by_name("feedState:0")
#tt = [n.name for n in tf.get_default_graph().as_graph_def().node]
#tt = [n.name for n in tf.get_default_graph().get_operations()]

generate_sample = graph.get_tensor_by_name("GenOut:0")

print("p1")
spread_product(Num_product_for_each, Num_product_category) #used for loop
print("p2")
feed_dict = {noise: currentState, is_training: False, feed_sub: currentState}

actionList = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
actionList = actionList.reshape(6, 6, 1, 1)
indices = np.zeros(BatchSize)


def Execute(v):

    random.seed()
    for u in range(v.shape[0]):
        idx_comp_max = 0
        idx_comp = 0
        idx_comp_sub = 0
        for r in range(v.shape[1]):
            if idx_comp_max <= v[u, r, 0, 0]:
                idx_comp_max = v[u, r, 0, 0]
                idx_comp_sub = idx_comp
                idx_comp = r
        #for r in range(v.shape[1]):
        #    if np.random.rand(1) < v[u, r, 0, 0]:
        #        comp[u, r, 0, 0] = 1
        #    else:
        #        comp[u, r, 0, 0] = 0
        #dist_2 = np.sum((actionList - v) ** 2, axis=1)
        #dist_2 = tf.reduce_sum((self.actionList - tf.matrix_transpose(v[u, :, :, :])) ** 2, reduction_indices=1)
        if idx_comp == 5:
            idx_comp = idx_comp_sub
        indices[u] = idx_comp#np.argmin(dist_2)

    _generator_nxt_state(indices)


def _generator_nxt_state(action_indices):

    for k in range(BatchSize):
        if np.random.rand(1) <= 0.2:
            action_indices[k] = np.random.randint(low=0, high=5, size=1, dtype=int)
        # nextState = currentState
        if action_indices[k] == 0:
            move_left()
        elif action_indices[k] == 1:
            move_right()
        elif action_indices[k] == 2:
            move_up()
        elif action_indices[k] == 3:
            move_down()
        elif action_indices[k] == 4:
            get()
        elif action_indices[k] == 5:
            #return
            cook()


######################### Auto Rewarding Test Code Line ###############################


rewards = [] #rewards[-1]
rewards_dist = [] #rewards[-1]
#des 0 : 1 ,2
#des 1 : 0, 3

def isSame(p1, p2):
    same = False
    k1 = p1.kind
    k2 = p2.kind
    idx1 = p1.idx_within_kind
    idx2 = p2.idx_within_kind
    if k1 == k2 and idx1 == idx2:
        same = True
    return same

RewardFactor = 2
def calcReward(desire, rewards, prodTarget1, prodTarget2, prodTarget_sub1, prodTarget_sub2):
    global ProductList

    rewardCalc = 0 #offset

    #panelty = 0 - No panelty for eating other ingr

    count = 0
    for v in ProductList:
        if not isSame(v, prodTarget1) and not isSame(v, prodTarget2) and not isSame(v, prodTarget_sub1) \
                and not isSame(v, prodTarget_sub2):
            count += 5
        else:
            if isSame(v, prodTarget1) or isSame(v, prodTarget2):
                rewardCalc -= RewardFactor*10  #v.quality_factor
            elif isSame(v, prodTarget_sub1) or isSame(v, prodTarget_sub2):
                rewardCalc -= RewardFactor*5  #v.quality_factor/2

    rewardCalc = rewardCalc + count
    return rewardCalc


def calcReward_dist(desire):
    global ProductList
    rewardCalc = 0

    isThere = False
    for v in ProductList:
        ptype = v.kind
        x = v.xCoord
        y = v.yCoord
        if desire == 0:
            if ptype == 1 or ptype == 2:
                rewardCalc += 10/(np.abs(x - AgentCoord[0]) + 1)
                rewardCalc += 10/(np.abs(y - AgentCoord[1]) + 1)
                isThere = True
        elif desire == 1:
            if ptype == 0 or ptype == 3:
                rewardCalc += 10/(np.abs(x - AgentCoord[0]) + 1)
                rewardCalc += 10/(np.abs(y - AgentCoord[1]) + 1)
                isThere = True
    if not isThere:
        rewardCalc  = 10
    return rewardCalc


def Make_agentBehave2(k, prodTarget1, prodTarget2, prodTarget_sub1, prodTarget_sub2, desire):

    global feed_dict
    global currentState
    global rewards_dist
    global rewards

    init = tf.global_variables_initializer()
    sess.run(init)

    feed_dict = {noise: currentState, is_training: False, feed_sub: currentState}
    #print(currentState.shape)
    #print(currentState[0, 0:StateDim, :, :].reshape(1, StateDim))
    result = sess.run(generate_sample, feed_dict=feed_dict)
    action = result[:, 0:ActionDim+2, :, :]
    #print(action.reshape(6, 1))
    Execute(action)
    #Draw Reward Chart Here

    rewards_dist.append(calcReward_dist(desire)) #Not Cumulative
    rewards.append(calcReward(desire, rewards, prodTarget1, prodTarget2, prodTarget_sub1, prodTarget_sub2)) #Cumulative

    plt.ylim(0, 60)
    plt.plot(rewards)
    plt.title('reward')
    plt.xlabel('iterations')
    plt.ylabel('reward-get')

    # plt.show()

    #plt.savefig('GAIL-rewards-500'+str(k))

    plt.plot(rewards_dist)
    plt.title('reward/reward-distance')
    plt.xlabel('iterations')
    plt.ylabel('reward-distance')
    # plt.show()
    plt.savefig('GAIL-rewards-dist-500'+str(k))


maxTrial = 5
maxItr = 100

for k in range(maxTrial):
    desire = state.Internal_state_Buffer[Maximum_inventory]

    prodTarget1 = None
    prodTarget2 = None

    prodTarget_sub1 = None
    prodTarget_sub2 = None

    if desire == 0:
        for v in ProductList:
            ptype = v.kind
            if ptype == 1:
                prodTarget1 = v
                if prodTarget1.quality_factor <= v.quality_factor:
                    prodTarget_sub1 = prodTarget1
                    prodTarget1 = v

            elif ptype == 2:
                prodTarget2 = v
                if prodTarget2.quality_factor <= v.quality_factor:
                    prodTarget_sub2 = prodTarget2
                    prodTarget2 = v

    elif desire == 1:
        for v in ProductList:
            ptype = v.kind
            if ptype == 0:
                prodTarget1 = v
                if prodTarget1.quality_factor <= v.quality_factor:
                    prodTarget_sub1 = prodTarget1
                    prodTarget1 = v
            elif ptype == 3:
                prodTarget2 = v
                if prodTarget2.quality_factor <= v.quality_factor:
                    prodTarget_sub2 = prodTarget2
                    prodTarget2 = v


    for i in range(maxItr):
        Make_agentBehave2(k, prodTarget1, prodTarget2, prodTarget_sub1, prodTarget_sub2, desire)
        wn.delay(20)
    cook()



















