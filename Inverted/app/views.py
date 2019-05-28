from django.shortcuts import render
from app.function import main

# Create your views here.

def index(request):

    return render(request, 'app/index.html')
    
def result(request):
    if request.method == 'POST':
        text = request.POST['querysearch']
        result, N, time = main.main_function(text)

    # frame = pd.DataFrame(list(score), columns=['DOC NO', 'FILENAME', 'SCORE TFIDF'])

    # hasil = frame.head(50).to_html(index=True, classes='table table-sm bordered table-condensed')

        content = {
            'result': result,
            'N': N,
            'time':time,
            'query' : text
        }

    return render(request, 'app/result.html', content)
