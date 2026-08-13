import base64
import json
import unittest

from scripts.aws.programming_brain_proxy import decode_relay_output, relay_command


class ProgrammingBrainProxyTests(unittest.TestCase):
    def test_relay_command_embeds_only_base64_request_material(self) -> None:
        body = b'{"message":"do not log this prompt"}'
        command = relay_command("/brain/chat", body)

        self.assertNotIn("do not log this prompt", command)
        self.assertIn(base64.b64encode(body).decode("ascii"), command)
        self.assertNotIn("shell=True", command)

    def test_decode_relay_output_uses_last_nonempty_line(self) -> None:
        expected = b'{"ok":true}'
        payload = json.dumps({
            "status": 201,
            "content_type": "application/json",
            "body": base64.b64encode(expected).decode("ascii"),
        })

        status, content_type, body = decode_relay_output(f"noise\n{payload}\n")

        self.assertEqual(status, 201)
        self.assertEqual(content_type, "application/json")
        self.assertEqual(body, expected)


if __name__ == "__main__":
    unittest.main()
