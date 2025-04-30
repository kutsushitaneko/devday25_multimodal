grant connect, ctxapp, dwrole, unlimited tablespace to devday25user;
grant execute on ctxsys.ctx_ddl to devday25user;
grant execute on DBMS_CLOUD_AI to devday25user;
  grant execute on DBMS_VECTOR to devday25user;
  grant execute on DBMS_VECTOR_CHAIN to devday25user;


BEGIN
    DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => '*',
        ace => xs$ace_type(
            privilege_list => xs$name_list('connect'),
            principal_name => 'devday25user',
            principal_type => xs_acl.ptype_db
        )
    );
END;
/