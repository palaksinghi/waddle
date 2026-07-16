import mujoco

model = mujoco.MjModel.from_xml_path("pusher.xml")

print("XML loaded successfully!")
print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)