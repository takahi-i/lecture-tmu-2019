from lecture_tmu_2019.utils import set_locale, get_unigram_from_text
from lecture_tmu_2019.ml import ReputationClassifier

set_locale()
classiier = ReputationClassifier()
classiier.fit()
result = classiier.predict(get_unigram_from_text(" ".split("tristar / 1 : 30 / 1997 / r ( language , violence , dennis rodman ) cast : jean-claude van damme ; " +
                            "mickey rourke ; dennis rodman ; natacha lindinger ; paul freeman director : tsui hark screenplay : "
                            "dan jakoby ; paul mones ripe with explosions , mass death and really weird hairdos , tsui hark's " +
                            "double team \" must be the result of a tipsy hollywood power lunch that decided jean-claude van" +
                            "damme needs another notch on his bad movie-bedpost and nba superstar dennis rodman should have an" +
                            "acting career . actually , in \" double team , \" neither's performance is all that bad . i've always" +
                            "been the one critic to defend van damme -- he possesses a high charisma level that some genre stars" +
                            "( namely steven seagal ) never aim for ; it's just that he's never made a movie so exuberantly witty since 1994's " +
                            "timecop . \" and rodman . . . well , he's pretty much rodman . he's extremely colorful , and therefore he pretty" +
                            "much fits his role to a t , even if the role is that of an ex-cia weapons expert . it's the story that needs some major" +
                            "work . van damme plays counter-terrorist operative jack quinn , who teams up with arms dealer yaz ( rodman )" +
                            " to rub out deadly gangster stavros ( mickey rourke , all beefy and weird-looking ) in an antwerp amusement park . "
                            "the job is botched when stavros' son gets killed in the gunfire , and quinn is taken off to an island known as \" the colony \" " +
                            "-- a think tank for soldiers \" too valuable to kill \" but \" too dangerous to set free . \" quinn escapes and tries" +
                            "to make it back home to his pregnant wife ( natacha lindinger ) , but stavros is out for revenge and kidnaps her . so , what's a" +
                            " kickboxing mercenary to do ? quinn looks up yaz and the two travel to rome so they can rescue the woman , kill stavros" +
                            " , save the world and do whatever else the screenplay requires them to do . with crazy , often eye-popping" +
                            " camera work by peter pau and rodman's lite brite locks , \" double team \" should be a mildly enjoyable guilty pleasure" +
                            ". but too much tries to happen in each frame , and the result is a movie that leaves you exhausted rather than exhilarated " +
                            ". the numerous action scenes are loud and headache-inducing and the frenetic pacing never slows" +
                            " down enough for us to care about what's going on in the movie . and much of what's going on is just wacky ."
                            " there's a whole segment devoted to net-surfing monks that i have yet to figure out . and the climax finds quinn" +
                            " going head-to-head with a tiger in the roman coliseum while yaz circles them on a motorcycle , trying to avoid running"  +
                            " over land mines and hold on to quinn's baby boy ( who's in a bomb equipped basket ) -- all this while stavro" +
                            " watches shirtless from the bleachers . did i mention \" double team \" is strange ? when it all comes down , " +
                            "this is just another rarely entertaining formula killathon , albeit one that feels no need to indulge in gratuitous" +
                            " profanity . rodman juices things up with his blatantly vibrant screen persona , though , leading up to a stunt " +
                            "where he kicks an opponent between the legs . but we didn't need \" double team \" to tell us he could do that , did we ? " +
                            "1997 jamie peck e-mail : <a href= \" mailto : jpeck1@gl . umbc . edu \" >jpeck1@gl . umbc . edu</a> visit the reel deal" +
                            " online : <a href= \" http : //www . gl . umbc . edu/~jpeck1/ \" >http : //www . gl . umbc . edu/~jpeck1/</a>\"")))
print(result)

