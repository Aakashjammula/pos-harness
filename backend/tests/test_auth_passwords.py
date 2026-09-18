from pos.auth.passwords import hash_password, verify_password


def test_hash_is_argon2id_and_not_the_password():
    digest = hash_password("correct horse battery staple")
    assert digest.startswith("$argon2id$")
    assert "correct horse battery staple" not in digest


def test_verify_accepts_the_right_password():
    digest = hash_password("s3cret")
    assert verify_password("s3cret", digest) is True


def test_verify_rejects_the_wrong_password():
    digest = hash_password("s3cret")
    assert verify_password("not-it", digest) is False


def test_two_hashes_of_the_same_password_differ():
    assert hash_password("same") != hash_password("same")


def test_verify_rejects_a_malformed_hash_instead_of_raising():
    assert verify_password("anything", "not-a-hash") is False
