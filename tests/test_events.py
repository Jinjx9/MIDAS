import unittest
from core.event_core import analyse,align_exact

class StrictEvents(unittest.TestCase):
    def test_multiple_wrong_classes_one_event(self):
        events,_,_=analyse([0]*5,[0,1,2,0,0],1)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['wrong_segment_count'],2)
        self.assertEqual(events[0]['extra_switches'],3)
    def test_boundary_and_whole_omission_not_internal(self):
        for p in ([1,0,0,0,0],[0,0,0,0,1],[1]*5):
            self.assertEqual(analyse([0]*5,p,1)[0],[])
    def test_no_event_and_exact_alignment(self):
        self.assertEqual(analyse([0]*5,[0]*5,1)[0],[])
        with self.assertRaises(ValueError):align_exact([0]*5,[0]*4,1,1)
        with self.assertRaises(ValueError):align_exact([0]*5,[0]*5,1,2)

if __name__=='__main__':unittest.main()
