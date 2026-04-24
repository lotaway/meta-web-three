package com.metawebthree.common.adapter;

import java.sql.Array;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;

public class SQLLongArrayHandler extends BaseTypeHandler<Long[]> {

    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, Long[] parameter, JdbcType jdbcType) throws SQLException {
        Connection conn = ps.getConnection();
        Array array = conn.createArrayOf("bigint", parameter);
        ps.setArray(i, array);
    }

    @Override
    public Long[] getNullableResult(ResultSet rs, String columnName) throws SQLException {
        return getResultFromArray(rs.getArray(columnName));
    }

    @Override
    public Long[] getNullableResult(ResultSet rs, int columnIndex) throws SQLException {
        return getResultFromArray(rs.getArray(columnIndex));
    }

    @Override
    public Long[] getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {
        return getResultFromArray(cs.getArray(columnIndex));
    }

    private Long[] getResultFromArray(java.sql.Array array) throws SQLException {
        if (array == null) {
            return null;
        }
        Long[] result = (Long[]) array.getArray();
        return result;
    }
}
