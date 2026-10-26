"""Release checks must reject partial exits, untouched rooms and old engine output."""
import copy
import unittest
from judge_gate import gate

class JudgeGateTest(unittest.TestCase):
    def setUp(self):
        self.house = {'rooms_poly': [{'id': 'a', 'poly': [[0,0],[2,0],[2,2],[0,2]]}, {'id':'b', 'poly':[[2,0],[4,0],[4,2],[2,2]]}]}
        self.frames = [{'t': 0, 'robots': [{'id':'robot_A','x':1,'y':1,'out':False}, {'id':'robot_B','x':3,'y':1,'out':False}]}, {'t':180, 'robots':[{'id':'robot_A','x':0,'y':-1,'out':True}, {'id':'robot_B','x':0,'y':-2,'out':True}]}]
        self.status = {'engine':{'git':'new'},'score':{'all_out':True,'robots_out':2,'false_confirmations':0}}
    def test_good_run(self):
        self.assertTrue(gate(self.house, self.frames, self.status, 'new', True)['passed'])
    def test_untouched_room(self):
        self.frames[0]['robots'][1]['x']=1
        self.assertEqual(gate(self.house,self.frames,self.status,'new',True)['missing_rooms'],['b'])
        self.assertFalse(gate(self.house,self.frames,self.status,'new',True)['passed'])
    def test_old_engine(self):
        self.assertFalse(gate(self.house,self.frames,self.status,'old',True)['passed'])
    def test_false_confirmation(self):
        self.status['score']['false_confirmations']=1
        self.assertFalse(gate(self.house,self.frames,self.status,'new',True)['passed'])
    def test_final_poses_override_an_optimistic_score(self):
        self.frames[-1]['robots'][1]['out']=False
        self.assertFalse(gate(self.house,self.frames,self.status,'new',True)['passed'])
    def test_main_houses_require_exit_and_false_gate_without_extra_room_gate(self):
        self.frames[0]['robots'][1]['x']=1
        self.assertTrue(gate(self.house,self.frames,self.status,'new',False)['passed'])

if __name__=='__main__':unittest.main()
