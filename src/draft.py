import pybullet as p
client_ID = p.connect(p.DIRECT)

while True:
    cuid = p.createCollisionShape(p.GEOM_BOX, halfExtents = [1, 1, 1])
    mass= 0
    body_id = p.createMultiBody(mass,cuid)

    p.removeBody(body_id)
    #p.removeCollisionShape(cuid)

    print(cuid,body_id)

