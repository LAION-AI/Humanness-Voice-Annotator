"""Contract checks for word preservation, time overlap and free-form captions."""
import copy
import unittest
from humanness_voice_annotator import compose_annotation, verify_script


def fixture():
    acoustic = {"id":"illustrative", "dur":4.2,"audio_sha256":"a"*64,
        "transcript":"I thought you might stay. Just for tonight.",
        "words":[{"w":w,"start":a,"end":b} for w,a,b in [
            ("I",0,.2),("thought",.2,.6),("you",.6,.9),("might",.9,1.2),("stay.",1.2,2.1),
            ("Just",3.0,3.3),("for",3.3,3.5),("tonight.",3.5,4.2)]],
        "sentences":[{"start":0,"end":2.1,"text":"I thought you might stay.","caption":"soft, restrained sadness"},
                     {"start":3.0,"end":4.2,"text":"Just for tonight.","caption":"quiet, hesitant"}]}
    moss = {"audio_sha256":"a"*64,"audio_duration_s":4.2,
        "voice_caption":{"prediction":"A softly spoken, hesitant voice."},
        "burst_captions":[{"start_s":2.5,"end_s":2.8,"caption":{"prediction":"A brief, breathy chuckle"}}],
        "locator_settings":{"threshold":.6,"min_duration_s":.08,"nms_overlap_fraction":.5,"event_cap":None}}
    return acoustic,moss


class FormatContract(unittest.TestCase):
    def test_standalone_pause_and_round_sentence_budget(self):
        a,m=fixture();r=compose_annotation(a,m)
        self.assertIn('(soft, restrained sadness; 2.10 seconds duration)',r['script'])
        self.assertIn('[0.40 seconds pause]',r['script'])
        self.assertIn('("A brief, breathy chuckle", 0.30 seconds)',r['script'])
        self.assertTrue(r['timing']['independent_verification']['verified'])
        self.assertEqual(r['burst_events'][0]['start_s'],2.5)

    def test_overlap_is_not_added_to_elapsed_budget(self):
        a,m=fixture();m['burst_captions'][0].update(start_s=.3,end_s=.6)
        r=compose_annotation(a,m)
        self.assertTrue(r['timing']['independent_verification']['verified'])
        self.assertAlmostEqual(r['timing']['independent_verification']['printed_budget_s'],4.2)
        self.assertIn('HVA_EVENT_0000',r['segments'][0]['inline_event_ids'])

    def test_free_form_punctuation_round_trips(self):
        a,m=fixture();text='A "caught" laugh (breathy), followed by a tiny sigh.\\ A second line\nends here.'
        m['burst_captions'][0]['caption']['prediction']=text
        r=compose_annotation(a,m)
        self.assertEqual(r['timing']['independent_verification']['parsed_bursts'][0]['caption'],text)
        self.assertEqual(r['burst_events'][0]['caption'],text)

    def test_overlapping_bursts_without_speech(self):
        a,m=fixture();a.update(dur=2.,words=[],sentences=[],transcript='')
        m.update(audio_duration_s=2.,burst_captions=[
            {'start_s':.2,'end_s':1.,'caption':{'prediction':'A gasp'}},
            {'start_s':.6,'end_s':1.4,'caption':{'prediction':'A sigh'}}])
        r=compose_annotation(a,m)
        self.assertTrue(r['timing']['independent_verification']['verified'])
        self.assertEqual(r['timing']['independent_verification']['burst_count'],2)
        self.assertEqual(r['timing']['independent_verification']['printed_budget_s'],2.)

    def test_rejects_audio_mismatch(self):
        a,m=fixture();m['audio_sha256']='b'*64
        with self.assertRaises(ValueError):compose_annotation(a,m)

    def test_rejects_unaligned_speech(self):
        a,m=fixture();a['words']=[]
        with self.assertRaises(ValueError):compose_annotation(a,m)

    def test_rejects_missing_endpoint(self):
        a,m=fixture();del m['burst_captions'][0]['end_s']
        with self.assertRaises(KeyError):compose_annotation(a,m)

    def test_verifier_detects_duplicated_word_and_wrong_time(self):
        a,m=fixture();r=compose_annotation(a,m)
        invalid=r['script'].replace('I thought','I I thought').replace('2.10 seconds duration','9.10 seconds duration')
        check=verify_script(invalid,r['aligned_transcript'],4.2,1)
        self.assertFalse(check['verified'])
        self.assertEqual(len(check['errors']),2)

    def test_deterministic_and_does_not_modify_inputs(self):
        a,m=fixture();before=copy.deepcopy((a,m))
        one=compose_annotation(a,m);two=compose_annotation(a,m)
        self.assertEqual(one['annotation_sha256'],two['annotation_sha256'])
        self.assertEqual((a,m),before)


if __name__=='__main__':unittest.main()
