package com.metawebthree.common.rpc;

import com.metawebthree.common.dto.UserRiskProfileDTO;
import com.metawebthree.common.dto.UserRiskProfileUpdateDTO;

public interface UserRiskProfileService {
    UserRiskProfileDTO getUserRiskProfile(Long userId);

    /**
     * Update user risk profile.
     * @param updateDTO the update data
     */
    void updateRiskProfile(UserRiskProfileUpdateDTO updateDTO);
}
