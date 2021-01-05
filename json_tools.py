# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json

def json_to_file(obj, filename):
    json_string = json.dumps(obj)
    fd = open(filename, "w")
    fd.write(json_string)
    fd.close()

def json_from_file(filename):
    fd = open(filename, "r")
    json_string = fd.read()
    fd.close()
    return json.loads(json_string)
