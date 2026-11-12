use gb_server::vision_ingress::{is_public_unicast, validate_addresses, validate_url};
#[test]
fn deny_all_special_targets_and_mixed_dns_answers() {
    for ip in [
        "0.1.2.3",
        "10.1.2.3",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.1.1",
        "172.16.0.1",
        "192.168.0.1",
        "192.0.0.1",
        "192.0.2.1",
        "198.18.0.1",
        "198.51.100.1",
        "203.0.113.1",
        "224.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",
        "64:ff9b:1::808:808",
        "2001::1",
        "2001:db8::1",
        "2002:808:808::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
    ] {
        assert!(!is_public_unicast(ip.parse().unwrap()), "{ip}");
    }
    for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
        assert!(is_public_unicast(ip.parse().unwrap()));
    }
    assert!(validate_addresses(vec!["8.8.8.8:443".parse().unwrap(), "127.0.0.1:443".parse().unwrap()]).is_err());
    for url in [
        "http://8.8.8.8/a",
        "https://127.0.0.1/a",
        "https://[::1]/a",
        "https://8.8.8.8:444/a",
        "https://u:p@example.com/a",
        "https://example.com/a#x",
        "file:///a",
        "https://2130706433/a",
    ] {
        assert!(validate_url(url).is_err(), "{url}");
    }
    assert!(validate_url("https://example.com:443/a").is_ok());
}
