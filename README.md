# CARLA-AU
A project on autonomous driving in CARLA simulator
## CARLA SIMULATOR -INTIALIAZATION

All commands should be run in the root __CARLA__ folder.
Commands should be executed via the __x64 Native Tools Command Prompt for VS 2019__. Open this by clicking the Windows key and searching for x64.
to navigate to carla root folder

> G:

>cd carla

>make launch

After __Unreal engine editor__ successfully, open command prompt 

>G:

> cd carla\PythonAPI\examples

> python generate_traffic.py

open another command prompt,navigate to examples folder as mentioned earlier*

>python automatic_control.py

(* this is done to improve the FPS (from 3 to 22))
