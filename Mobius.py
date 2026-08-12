import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from matplotlib import pyplot as plt
    from cmath import sin

    return (plt,)


@app.cell
def _():
    def calcMobius(z1,w1,z2,w2,z3,w3):
      zw1=z1*w1
      zw2=z2*w2
      zw3=z3*w3
      dz12=z1-z2
      dz23=z2-z3
      dz31=z3-z1 
      dw12=w1-w2
      dw23=w2-w3
      dw31=w3-w1 
      a=zw1*dw23 + zw2*dw31 + zw3*dw12
      b=zw1*(z2*w3-z3*w2) + zw2*(z3*w1-z1*w3) + zw3*(z1*w2-z2*w1)
      c=-(w1*dz23 + w2*dz31 + w3*dz12)
      d=zw1*dz23 + zw2*dz31 + zw3*dz12
      if abs(c)==0:
        s=1/d #keep unity denominator if only scaling and rotating 
      else:
        s=abs(a*d-b*c)**-0.5
        if a.real<0:
          s*=-1 # keep a positive 
      return a*s,b*s,c*s,d*s
  
    def transformMobius(a=1,b=0,c=0,d=1):
        return  lambda p:(a*p+b)/(c*p+d)

    z1,z2,z2=-1,0,1
    w1,w2,w3=-1+1j,0+1j,1+1j

    print(f'{calcMobius(-1,-1+1j, 0, 0+1j, 1,1+1j)=}')
    print(f'{calcMobius(-1,-1j, 0,0, 1,1j)=}')
    print(f'{calcMobius(-1,1j, 0,0, 1,-1j)=}')
    print(f'{calcMobius(-1,2, 0,0, 1,-2)=}')
    print(f'{calcMobius(-1,-2, 0,0, 1,2)=}')
    return calcMobius, transformMobius


@app.cell
def _(calcMobius, plt, transformMobius):
    fig=plt.figure()
    ax=fig.add_subplot()
    xy=[(i-5)/5 for i in range(11)]
    xy10=[(i-50)/50 for i in range(101)]
    b=.1j
    for t,style in ((transformMobius(),'k'),
                    (transformMobius(a=1.2),'b'),
                    (transformMobius(a=1+1j),'r'),
      (transformMobius(*calcMobius(-1,-1.5-b, 0,b, 1,1.5-b)),'g')):
      for x in xy:
        ax.plot(*list(zip(*[(p.real,p.imag) for x_,y in zip([x]*101,xy10) for p in (t(x_+1j*y),)])) ,style)
      for y in xy:
        ax.plot(*list(zip(*[(p.real,p.imag) for x_,y in zip(xy10,[y]*101) for p in (t(x_+1j*y),)])) ,style)
    ax.set_aspect('equal')
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
