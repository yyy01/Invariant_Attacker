html_template = """
<!DOCTYPE html>  
<html lang="zh">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Word List with Backgrounds</title>  
    <style>  
        .word {{
            padding: 1px;  
            margin: 1px;  
            border-radius: 0px;  
            color: black;
            font-family: "Times New Roman";
            display: inline-block;
        }}  
        #wordList1 {{  
            border: 1.5px solid black;
            padding: 2px;
            margin: 20px;
            max-width: 412px;
            max-height: 300px;
        }}  
    </style>  
</head>  
<body>  

<div id="wordList1"></div>  

<script>  
    const data1 = {DATA1};

    function getColor(weight) {{   
        const normalized = 1 - Math.min(weight, 1.0);
        let p = 0.92;
        const red = 255;
        const green = 60;
        const blue = 60;
        let alpha = 1-normalized*0.9;
        if (normalized > p){{
            alpha = 0.1;
        }}
        let p_high = 0.85;
        let p_low = 0.6;
        if (normalized > p_low && normalized < p_high){{
            alpha = 0.5;
        }}
        if (normalized < p_low){{
            alpha = 0.9;
        }}
        return `rgba(${{red}}, ${{green}}, ${{blue}}, ${{alpha}})`;
    }}  

    // Painting Wordlists
    const wordListDiv1 = document.getElementById('wordList1');  
    data1.words.forEach((word, index) => {{  
        const weight = data1.weights[index];  
        const wordSpan = document.createElement('span');  
        wordSpan.className = 'word';  
        wordSpan.style.backgroundColor = getColor(weight);  
        wordSpan.textContent = word;  
        wordListDiv1.appendChild(wordSpan);  
    }});
</script>  

</body>  
</html>
"""