def function( password = "" ):
    if len(password) < 2:
        return 0
 
    vowels = set( ['a', 'e', 'i', 'o', 'u'] )
    password = password.lower()
    if password[0] in vowels:
        return 1 + function( password[ 1 : : 2 ] )
    
    return function( password[ 1 : ] ) + len( password[ 2 : 5 ] )
    
print( function( "APple" ) + function( "lemming" ) )