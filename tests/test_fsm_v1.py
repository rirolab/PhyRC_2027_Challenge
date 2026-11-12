import unittest
from phase2_reorientation.fsm_v1 import Evidence, Phase, ReorientationFSM, Settings


def ev(t, **kw):
    return Evidence(time_s=t,valid_cameras=True,**kw)

class TestFSM(unittest.TestCase):
    def test_gripper_intent_is_not_grasp_evidence(self):
        f=ReorientationFSM()
        self.assertEqual(f.step(ev(0,both_grippers_commanded_closed=True)),Phase.CHECK_GRASP)
        self.assertEqual(f.step(ev(.05,both_grippers_commanded_closed=True)),Phase.CHECK_GRASP)
    def test_complete_explicit_evidence(self):
        f=ReorientationFSM(Settings(min_verified_duration_s=.2))
        samples=[ev(0),ev(.05,grasp_visually_supported=True),
            ev(.10,grasp_visually_supported=True,orientation_known=True,plan_safe=True),
            ev(.15,grasp_visually_supported=True,plan_safe=True,execution_started=True),
            ev(.20,grasp_visually_supported=True,plan_safe=True,execution_completed=True),
            ev(.25,grasp_visually_supported=True,target_verified=True),
            ev(.50,grasp_visually_supported=True,target_verified=True)]
        for x in samples:f.step(x)
        self.assertEqual(f.phase,Phase.SUCCESS)
    def test_intermittent_target_does_not_pass(self):
        f=ReorientationFSM(Settings(min_verified_duration_s=.2))
        for x in [ev(0), ev(.05,grasp_visually_supported=True),
            ev(.10,grasp_visually_supported=True,orientation_known=True,plan_safe=True),
            ev(.15,grasp_visually_supported=True,plan_safe=True,execution_started=True),
            ev(.2,grasp_visually_supported=True,plan_safe=True,execution_completed=True),
            ev(.25,grasp_visually_supported=True,target_verified=True),
            ev(.30,grasp_visually_supported=True,target_verified=None),
            ev(.50,grasp_visually_supported=True,target_verified=True)]:f.step(x)
        self.assertEqual(f.phase,Phase.VERIFY)
    def test_plan_reject_to_recovery(self):
        f=ReorientationFSM()
        f.step(ev(0));f.step(ev(.05,grasp_visually_supported=True))
        self.assertEqual(f.step(ev(.1,grasp_visually_supported=True,plan_safe=False)),Phase.RECOVER)
    def test_vision_lost_during_execution(self):
        f=ReorientationFSM()
        f.step(ev(0));f.step(ev(.05,grasp_visually_supported=True))
        f.step(ev(.1,grasp_visually_supported=True,orientation_known=True,plan_safe=True))
        self.assertEqual(f.step(Evidence(time_s=.15,valid_cameras=False)),Phase.RECOVER)
    def test_no_fake_success_from_hdf5_only(self):
        f=ReorientationFSM()
        for i in range(10):
            f.step(ev(.05*i,both_grippers_commanded_closed=True))
        self.assertEqual(f.phase,Phase.CHECK_GRASP)
    def test_time_order_rejected(self):
        f=ReorientationFSM();f.step(ev(1))
        with self.assertRaises(ValueError):f.step(ev(1))
    def test_timeout(self):
        f=ReorientationFSM(Settings(phase_timeout_s=.1))
        f.step(ev(0))
        self.assertEqual(f.step(ev(.2)),Phase.RECOVER)

if __name__=='__main__':unittest.main()
