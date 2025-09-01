package com.metawebthree.common.adapter;

import java.sql.Array;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;

public class SQLShortArrayHandler extends BaseTypeHandler<Short[]> {

    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, Short[] parameter, JdbcType jdbcType) throws SQLException {
        Connection conn = ps.getConnection();
        Array array = conn.createArrayOf("smallint", parameter);
        ps.setArray(i, array);
    }

    @Override
    public Short[] getNullableResult(ResultSet rs, String columnName) throws SQLException {
        return getResultFromArray(rs.getArray(columnName));
    }

    @Override
    public Short[] getNullableResult(ResultSet rs, int columnIndex) throws SQLException {
        return getResultFromArray(rs.getArray(columnIndex));
    }

    @Override
    public Short[] getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {
        return getResultFromArray(cs.getArray(columnIndex));
    }

    private Short[] getResultFromArray(java.sql.Array array) throws SQLException {
        if (array == null) {
            return null;
        }
        Short[] result = (Short[]) array.getArray();
        return result;
    }
}