package com.metawebthree.repository;

import com.metawebthree.entity.UserKYC;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserKYCRepository extends JpaRepository<UserKYC, Long> {
    
    Optional<UserKYC> findByUserId(Long userId);
    
    Optional<UserKYC> findByUserIdAndStatus(Long userId, UserKYC.KYCStatus status);
    
    List<UserKYC> findByStatus(UserKYC.KYCStatus status);
    
    List<UserKYC> findByLevel(UserKYC.KYCLevel level);
    
    @Query("SELECT kyc FROM UserKYC kyc WHERE kyc.userId = :userId AND kyc.status = 'APPROVED' ORDER BY kyc.level DESC LIMIT 1")
    Optional<UserKYC> findHighestApprovedLevelByUserId(@Param("userId") Long userId);
    
    @Query("SELECT kyc FROM UserKYC kyc WHERE kyc.userId = :userId ORDER BY kyc.createdAt DESC LIMIT 1")
    Optional<UserKYC> findLatestByUserId(@Param("userId") Long userId);
    
    boolean existsByUserIdAndStatus(Long userId, UserKYC.KYCStatus status);
    
    @Query("SELECT COUNT(kyc) FROM UserKYC kyc WHERE kyc.status = 'PENDING'")
    Long countPendingKYC();
    
    @Query("SELECT kyc FROM UserKYC kyc WHERE kyc.status = 'PENDING' ORDER BY kyc.createdAt ASC")
    List<UserKYC> findPendingKYC();
} 