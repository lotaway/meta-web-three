package com.metawebthree.common.rpc;

import com.metawebthree.common.dto.UserRiskProfileDTO;

public interface UserRiskProfileService {
    UserRiskProfileDTO getUserRiskProfile(Long userId);
}
