from tool import TextQualityEvaluator

evaluator = TextQualityEvaluator()

text = "What if it be a poison, which the friar subtly hath minister'd to have me dead, lest in this marriage he should be dishonour'd, because he married me before to Romeo? I fear it is: and yet, methinks, it should not, for he hath still been tried a holy man. How if, when I am laid into the tomb, I wake before the time that Romeo come to redeem me? There's a fearful point! Shall I not, then, be stifled in the vault, to whose foul mouth no healthsome air breathes in, and there die strangled ere my Romeo comes? Or, if I live, is it not very like, the horrible conceit of death and night, together with the terror of the place, as in a vault, an ancient receptacle, where, for these many hundred years, the bones of all my buried ancestors are packed: where bloody Tybalt, yet but green in earth, lies festering in his shroud; where, as they say, at some hours in the night spirits resort; alack, alack, is it not like that I, so early waking, what with loathsome smells, and shrieks like mandrakes' torn out of the earth, that living mortals, hearing them, run mad: o, if I wake, shall I not be distraught, environed with all these hideous fears?"

result = evaluator.evaluate_text(text, "adults with higher education")

print(result)
