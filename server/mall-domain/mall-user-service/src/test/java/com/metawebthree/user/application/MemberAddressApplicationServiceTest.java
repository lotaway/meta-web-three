package com.metawebthree.user.application;

import com.metawebthree.user.domain.model.MemberAddress;
import com.metawebthree.user.domain.repository.MemberAddressRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class MemberAddressApplicationServiceTest {

    private MemberAddressRepository repository;
    private MemberAddressApplicationService service;

    @BeforeEach
    void setUp() {
        repository = mock(MemberAddressRepository.class);
        service = new MemberAddressApplicationService(repository);
    }

    private MemberAddress address(Long memberId, String name, String phone, boolean isDefault) {
        MemberAddress address = new MemberAddress();
        address.setMemberId(memberId);
        address.setName(name);
        address.setPhoneNumber(phone);
        address.setDefaultStatus(isDefault);
        return address;
    }

    @Test
    void addAddressSavesAndClearsDefaultWhenMarkingDefault() {
        service.addAddress(address(7L, "Alice", "13800000000", true));

        verify(repository).clearDefaultStatus(7L);
        verify(repository).save(any(MemberAddress.class));
    }

    @Test
    void addAddressDoesNotClearDefaultForRegularAddress() {
        service.addAddress(address(7L, "Alice", "13800000000", false));

        verify(repository, never()).clearDefaultStatus(any());
        verify(repository).save(any(MemberAddress.class));
    }

    @Test
    void addAddressRejectsMissingName() {
        assertThrows(IllegalArgumentException.class,
                () -> service.addAddress(address(7L, null, "13800000000", false)));
        verify(repository, never()).save(any());
    }

    @Test
    void addAddressRejectsMissingPhone() {
        assertThrows(IllegalArgumentException.class,
                () -> service.addAddress(address(7L, "Alice", null, false)));
        verify(repository, never()).save(any());
    }

    @Test
    void updateAddressRequiresId() {
        MemberAddress address = address(7L, "Alice", "13800000000", false);

        assertThrows(IllegalArgumentException.class, () -> service.updateAddress(address));
        verify(repository, never()).save(any());
    }

    @Test
    void updateAddressSavesWhenIdPresent() {
        MemberAddress address = address(7L, "Alice", "13800000000", false);
        address.setId(1L);

        service.updateAddress(address);

        verify(repository).save(address);
    }

    @Test
    void listAddressesRequiresMemberId() {
        assertThrows(IllegalArgumentException.class, () -> service.listAddresses(null));
    }

    @Test
    void getAddressByIdReturnsNullWhenMissing() {
        when(repository.findById(99L)).thenReturn(Optional.empty());

        assertNull(service.getAddressById(99L));
    }

    @Test
    void getAddressByIdReturnsStoredAddress() {
        MemberAddress stored = address(7L, "Alice", "13800000000", false);
        stored.setId(1L);
        when(repository.findById(1L)).thenReturn(Optional.of(stored));

        assertEquals("Alice", service.getAddressById(1L).getName());
    }

    @Test
    void removeAddressRequiresId() {
        assertThrows(IllegalArgumentException.class, () -> service.removeAddress(null));
    }

    @Test
    void listAddressesReturnsRepositoryResult() {
        when(repository.findByMemberId(7L)).thenReturn(List.of(address(7L, "Alice", "13800000000", false)));

        assertEquals(1, service.listAddresses(7L).size());
    }
}
