package com.metawebthree.user.application;

import com.metawebthree.user.domain.model.MemberAddress;
import com.metawebthree.user.domain.repository.MemberAddressRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class MemberAddressApplicationService {

    private final MemberAddressRepository addressRepository;

    public void addAddress(MemberAddress address) {
        validateAddress(address);
        handleDefaultLogic(address);
        addressRepository.save(address);
    }

    public void updateAddress(MemberAddress address) {
        validateId(address.getId());
        validateAddress(address);
        handleDefaultLogic(address);
        addressRepository.save(address);
    }

    public List<MemberAddress> listAddresses(Long memberId) {
        validateId(memberId);
        return addressRepository.findByMemberId(memberId);
    }

    public void removeAddress(Long id) {
        validateId(id);
        addressRepository.deleteById(id);
    }

    private void handleDefaultLogic(MemberAddress address) {
        if (address.isDefaultStatus()) {
            addressRepository.clearDefaultStatus(address.getMemberId());
        }
    }

    private void validateAddress(MemberAddress addr) {
        if (addr.getName() == null || addr.getPhoneNumber() == null) {
            throw new IllegalArgumentException("Address name and phone number are required");
        }
    }

    private void validateId(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("Invalid address or member identity");
        }
    }
}
