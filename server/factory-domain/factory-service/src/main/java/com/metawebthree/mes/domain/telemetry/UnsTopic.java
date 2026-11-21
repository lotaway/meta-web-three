package com.metawebthree.mes.domain.telemetry;

import java.util.Objects;

public record UnsTopic(
    String enterprise,
    String site,
    String area,
    String line,
    String cell,
    String device,
    String tag
) {
    public UnsTopic {
        Objects.requireNonNull(enterprise, "enterprise");
        Objects.requireNonNull(site, "site");
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(tag, "tag");
    }

    public String asTopic() {
        return String.join("/",
            segment(enterprise), segment(site), segment(area),
            segment(line), segment(cell), segment(device), "Tag", segment(tag));
    }

    private static String segment(String value) {
        return value == null || value.isBlank() ? "_" : value;
    }
}
