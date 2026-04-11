package com.metawebthree.user.infrastructure.persistence;

import com.metawebthree.user.domain.model.MemberAddress;
import com.metawebthree.user.domain.repository.MemberAddressRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Repository
public class MemberAddressRepositoryImpl implements MemberAddressRepository {

    private final NamedParameterJdbcTemplate jdbcTemplate;

    @Autowired
    public MemberAddressRepositoryImpl(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = new NamedParameterJdbcTemplate(jdbcTemplate);
    }

    @Override
    @Transactional
    public void save(MemberAddress address) {
        String sql = "INSERT INTO member_address (member_id, name, phone_number, default_status, post_code, province, city, region, detail_address) "
                +
                "VALUES (:memberId, :name, :phoneNumber, :defaultStatus, :postCode, :province, :city, :region, :detailAddress)";
        SqlParameterSource params = new MapSqlParameterSource()
                .addValue("memberId", address.getMemberId())
                .addValue("name", address.getName())
                .addValue("phoneNumber", address.getPhoneNumber())
                .addValue("defaultStatus", address.isDefaultStatus())
                .addValue("postCode", address.getPostCode())
                .addValue("province", address.getProvince())
                .addValue("city", address.getCity())
                .addValue("region", address.getRegion())
                .addValue("detailAddress", address.getDetailAddress());
        jdbcTemplate.update(sql, params);
    }

    @Override
    @Transactional
    public void update(MemberAddress address) {
        String sql = "UPDATE member_address SET member_id = :memberId, name = :name, phone_number = :phoneNumber, default_status = :defaultStatus, post_code = :postCode, "
                +
                "province = :province, city = :city, region = :region, detail_address = :detailAddress WHERE id = :id";
        SqlParameterSource params = new MapSqlParameterSource()
                .addValue("id", address.getId())
                .addValue("memberId", address.getMemberId())
                .addValue("name", address.getName())
                .addValue("phoneNumber", address.getPhoneNumber())
                .addValue("defaultStatus", address.isDefaultStatus())
                .addValue("postCode", address.getPostCode())
                .addValue("province", address.getProvince())
                .addValue("city", address.getCity())
                .addValue("region", address.getRegion())
                .addValue("detailAddress", address.getDetailAddress());
        jdbcTemplate.update(sql, params);
    }

    @Override
    public MemberAddress findById(Long id) {
        String sql = "SELECT * FROM member_address WHERE id = :id";
        SqlParameterSource params = new MapSqlParameterSource("id", id);
        List<MemberAddress> results = jdbcTemplate.query(sql, params,
                BeanPropertyRowMapper.newInstance(MemberAddress.class));
        return results.isEmpty() ? null : results.get(0);
    }

    @Override
    public List<MemberAddress> findByMemberId(Long memberId) {
        String sql = "SELECT * FROM member_address WHERE member_id = :memberId";
        SqlParameterSource params = new MapSqlParameterSource("memberId", memberId);
        return jdbcTemplate.query(sql, params, BeanPropertyRowMapper.newInstance(MemberAddress.class));
    }

    @Override
    @Transactional
    public void delete(Long id) {
        String sql = "DELETE FROM member_address WHERE id = :id";
        SqlParameterSource params = new MapSqlParameterSource("id", id);
        jdbcTemplate.update(sql, params);
    }

    @Override
    @Transactional
    public void clearDefaultStatus(Long memberId) {
        String sql = "UPDATE member_address SET default_status = false WHERE member_id = :memberId AND default_status = true";
        SqlParameterSource params = new MapSqlParameterSource("memberId", memberId);
        jdbcTemplate.update(sql, params);
    }
}
